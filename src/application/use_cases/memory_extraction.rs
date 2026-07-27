//! Session memory extraction.
//!
//! Flow (single LLM call with one format-recovery retry):
//!
//! 1. **Prefetch** — embed a compact query built from the transcript and
//!    fetch the most similar existing memories, so the model can merge new
//!    information into them instead of creating duplicates.
//! 2. **Extract** — send the extraction instruction + memory-kind schemas +
//!    existing memories + conversation to the (small) chat model, which
//!    returns one JSON object of upsert/delete operations.
//! 3. **Apply** — validate and normalize the operations, then write them to
//!    the memory repository with fresh embeddings.

use std::sync::Arc;

use serde::Deserialize;
use tracing::{debug, warn};

use openai_rs::ChatClient;

use crate::application::interfaces::{Embedder, MemoryRepository};
use crate::application::use_cases::memory_extraction_prompt as prompt;
use crate::application::use_cases::memory_support::{unix_now, upsert_preserving_identity};
use crate::domain::{DomainError, MemoryItem, MemoryKind, MemoryOperation, SessionTranscript};

/// How many existing memories are prefetched into the extraction context.
const PREFETCH_LIMIT: usize = 8;

/// Upper bound on operations applied from a single extraction, as a guard
/// against a runaway model flooding the store with noise.
const MAX_OPERATIONS_PER_RUN: usize = 24;

/// Maximum length of a normalized item name.
const MAX_NAME_CHARS: usize = 64;

/// Outcome of one extraction run.
#[derive(Debug, Default)]
pub struct ExtractionReport {
    /// Operations that were applied, in order.
    pub applied: Vec<MemoryOperation>,
    /// Operations that were skipped, with the reason.
    pub skipped: Vec<(MemoryOperation, String)>,
}

impl ExtractionReport {
    /// How many distinct items the run actually wrote.
    ///
    /// Counts unique `(kind, name, project)` identities rather than upsert
    /// operations: a model that emits the same item several times in one
    /// response overwrites its own earlier write, leaving a single row, and
    /// reporting "3 items written" for one stored memory is a lie.
    pub fn items_written(&self) -> usize {
        self.applied
            .iter()
            .filter_map(|op| match op {
                MemoryOperation::Upsert {
                    kind,
                    name,
                    project,
                    ..
                } => Some((*kind, name.as_str(), project.as_deref())),
                MemoryOperation::Delete { .. } => None,
            })
            .collect::<std::collections::HashSet<_>>()
            .len()
    }
}

/// JSON shape the extraction model must return.
#[derive(Debug, Deserialize)]
struct ExtractionOutput {
    #[serde(default)]
    preferences: Vec<RawItem>,
    #[serde(default)]
    experiences: Vec<RawItem>,
    #[serde(default)]
    skills: Vec<RawItem>,
    #[serde(default)]
    facts: Vec<RawItem>,
    #[serde(default)]
    delete: Vec<RawDelete>,
}

#[derive(Debug, Deserialize)]
struct RawItem {
    #[serde(default)]
    name: String,
    #[serde(default)]
    content: String,
    /// Optional project marker: `true`/the project name marks this item as
    /// specific to the session's project; absent/`false` means global. The
    /// model is told to set it only for project-specific insights.
    #[serde(default)]
    project_specific: bool,
}

#[derive(Debug, Deserialize)]
struct RawDelete {
    #[serde(default)]
    kind: String,
    #[serde(default)]
    name: String,
}

pub struct MemoryExtractionUseCase {
    chat_client: Arc<dyn ChatClient>,
    memory_repo: Arc<dyn MemoryRepository>,
    embedding_service: Arc<Embedder>,
}

impl MemoryExtractionUseCase {
    pub fn new(
        chat_client: Arc<dyn ChatClient>,
        memory_repo: Arc<dyn MemoryRepository>,
        embedding_service: Arc<Embedder>,
    ) -> Self {
        Self {
            chat_client,
            memory_repo,
            embedding_service,
        }
    }

    /// Run extraction over a transcript and apply the resulting operations.
    #[tracing::instrument(skip_all, fields(session_id = %transcript.id))]
    pub async fn execute(
        &self,
        transcript: &SessionTranscript,
    ) -> Result<ExtractionReport, DomainError> {
        let existing = self.prefetch(transcript).await;
        let operations = self.extract(transcript, &existing).await?;
        self.apply(transcript, operations).await
    }

    /// Fetch existing memories related to this conversation so the model can
    /// update them in place. Prefetch failures degrade to "no context" rather
    /// than failing the import.
    async fn prefetch(&self, transcript: &SessionTranscript) -> Vec<MemoryItem> {
        if self.embedding_service.embeddings_enabled() {
            let query = prompt::prefetch_query(transcript);
            if query.is_empty() {
                return Vec::new();
            }
            match self.embedding_service.embed_query(&query).await {
                Ok(vector) => {
                    // Prefetch within the session's project (its items +
                    // globals) so merging happens against memories that are
                    // actually relevant to this project/namespace.
                    match self
                        .memory_repo
                        .search_semantic(
                            &vector,
                            None,
                            transcript.project.as_deref(),
                            PREFETCH_LIMIT,
                        )
                        .await
                    {
                        Ok(results) => return results.into_iter().map(|(item, _)| item).collect(),
                        Err(e) => warn!("memory prefetch search failed: {e}"),
                    }
                }
                Err(e) => warn!("memory prefetch embedding failed: {e}"),
            }
        }
        // No embeddings: surface the most recent items instead so merging
        // still has a chance to happen. Filter to the transcript's project
        // (global memories plus its project/namespace items) before the limit.
        match self.memory_repo.list_items(None).await {
            Ok(items) => {
                let mut filtered: Vec<MemoryItem> = items
                    .into_iter()
                    .filter(
                        |item| match (item.project(), transcript.project.as_deref()) {
                            (None, _) => true,
                            (Some(item_project), Some(project)) => item_project == project,
                            (Some(_), None) => false,
                        },
                    )
                    .collect();
                filtered.truncate(PREFETCH_LIMIT);
                filtered
            }
            Err(e) => {
                warn!("memory prefetch list failed: {e}");
                Vec::new()
            }
        }
    }

    /// Call the extraction model and parse its JSON output, retrying once
    /// with a format-correction message when parsing fails.
    ///
    /// The request is sent via [`ChatClient::complete_json`] with the extraction
    /// schema, so backends that support structured decoding return
    /// schema-conforming JSON directly. Backends without it fall back to
    /// free-form output; the tolerant [`parse_operations`] (fence stripping +
    /// escape repair) and the one-shot format retry cover that case.
    async fn extract(
        &self,
        transcript: &SessionTranscript,
        existing: &[MemoryItem],
    ) -> Result<Vec<MemoryOperation>, DomainError> {
        let system = prompt::system_prompt();
        let user = prompt::user_prompt(transcript, existing);
        let schema = extraction_schema();

        let project = transcript.project.as_deref();
        let response = self
            .chat_client
            .complete_json(&system, &user, "memory_extraction", &schema)
            .await?;
        match parse_operations(&response, project) {
            Ok(ops) => Ok(ops),
            Err(first_err) => {
                debug!("extraction output unparseable, retrying once: {first_err}");
                let retry_user = format!("{user}\n\n{}", prompt::format_retry_prompt());
                let response = self
                    .chat_client
                    .complete_json(&system, &retry_user, "memory_extraction", &schema)
                    .await?;
                parse_operations(&response, project).map_err(|e| {
                    DomainError::parse(format!(
                        "extraction model returned unparseable output twice: {e}"
                    ))
                })
            }
        }
    }

    /// Apply validated operations to the memory store.
    async fn apply(
        &self,
        transcript: &SessionTranscript,
        operations: Vec<MemoryOperation>,
    ) -> Result<ExtractionReport, DomainError> {
        let mut report = ExtractionReport::default();
        let now = unix_now();

        for op in operations.into_iter() {
            if report.applied.len() >= MAX_OPERATIONS_PER_RUN {
                report
                    .skipped
                    .push((op, "operation limit reached".to_string()));
                continue;
            }
            match op {
                MemoryOperation::Upsert {
                    kind,
                    ref name,
                    ref content,
                    ref project,
                } => {
                    upsert_preserving_identity(
                        self.memory_repo.as_ref(),
                        self.embedding_service.as_ref(),
                        kind,
                        name,
                        content,
                        project.clone(),
                        Some(&transcript.id),
                        now,
                    )
                    .await?;
                    report.applied.push(op);
                }
                MemoryOperation::Delete { kind, ref name } => {
                    // The model can only be referring to something it saw in
                    // its prefetch context: this session's project, or a
                    // global. Resolve within that scope so a delete never
                    // reaches a same-named memory owned by another project.
                    let candidates = self.memory_repo.find_items_named(kind, name).await?;
                    let target = candidates
                        .iter()
                        .find(|item| item.project() == transcript.project.as_deref())
                        .or_else(|| candidates.iter().find(|item| item.project().is_none()));
                    match target {
                        Some(item) => {
                            self.memory_repo.delete_item_by_id(item.id()).await?;
                            report.applied.push(op);
                        }
                        None => report.skipped.push((op, "item not found".to_string())),
                    }
                }
            }
        }
        Ok(report)
    }
}

/// JSON Schema for [`ExtractionOutput`], passed to structured-output backends
/// so the model's response is grammar-constrained to the exact shape we parse.
/// Kept in sync with the `ExtractionOutput` / `RawItem` / `RawDelete` structs.
fn extraction_schema() -> serde_json::Value {
    let item = serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "content": { "type": "string" },
            "project_specific": { "type": "boolean" }
        },
        "required": ["name", "content", "project_specific"],
        "additionalProperties": false
    });
    let item_array = serde_json::json!({ "type": "array", "items": item });
    serde_json::json!({
        "type": "object",
        "properties": {
            "preferences": item_array,
            "experiences": item_array,
            "skills": item_array,
            "facts": item_array,
            "delete": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["preference", "experience", "skill", "fact"]
                        },
                        "name": { "type": "string" }
                    },
                    "required": ["kind", "name"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["preferences", "experiences", "skills", "facts", "delete"],
        "additionalProperties": false
    })
}

/// Parse the model's JSON response into validated, normalized operations.
///
/// Tolerates surrounding prose or a markdown fence by extracting the first
/// balanced top-level JSON object. `project` is the session's project name; an
/// item the model flagged `project_specific` is assigned to it (items stay
/// global when the session had no known project, since there is nothing to
/// assign them to).
fn parse_operations(
    response: &str,
    project: Option<&str>,
) -> Result<Vec<MemoryOperation>, DomainError> {
    let json = extract_json_object(response)
        .ok_or_else(|| DomainError::parse("no JSON object found in extraction output"))?;
    // Small local models routinely emit markdown content with invalid JSON
    // escapes (`\_`, `\(`, a raw newline inside a string, a stray trailing
    // `\`). Try strict parsing first so well-formed output is untouched, then
    // fall back to a repaired copy before giving up.
    let output: ExtractionOutput = match serde_json::from_str(json) {
        Ok(output) => output,
        Err(strict_err) => {
            let repaired = repair_json_string_escapes(json);
            serde_json::from_str(&repaired)
                .map_err(|_| DomainError::parse(format!("invalid extraction JSON: {strict_err}")))?
        }
    };

    let mut operations = Vec::new();
    let groups = [
        (MemoryKind::Preference, output.preferences),
        (MemoryKind::Experience, output.experiences),
        (MemoryKind::Skill, output.skills),
        (MemoryKind::Fact, output.facts),
    ];
    for (kind, items) in groups {
        for item in items {
            let Some(name) = normalize_name(&item.name) else {
                continue;
            };
            let content = item.content.trim();
            if content.is_empty() {
                continue;
            }
            // Assign a project only when the model marked the item
            // project-specific AND we actually know the project; otherwise
            // keep it global.
            let item_project = if item.project_specific {
                project.map(str::to_string)
            } else {
                None
            };
            operations.push(MemoryOperation::Upsert {
                kind,
                name,
                content: content.to_string(),
                project: item_project,
            });
        }
    }
    for del in output.delete {
        let Some(kind) = MemoryKind::parse(&del.kind) else {
            continue;
        };
        let Some(name) = normalize_name(&del.name) else {
            continue;
        };
        operations.push(MemoryOperation::Delete { kind, name });
    }
    Ok(operations)
}

/// Repair invalid backslash escapes inside JSON string literals.
///
/// Small local models frequently emit markdown content with escapes that are
/// valid in Markdown but invalid in JSON — `\_`, `\(`, `\<`, a trailing `\`,
/// or raw control characters (a literal newline/tab) inside a string. Strict
/// `serde_json` rejects all of these. This walks the text tracking string
/// context and, inside strings, passes valid JSON escapes through untouched
/// while escaping anything else so the result parses. Text outside strings is
/// left exactly as-is.
pub(crate) fn repair_json_string_escapes(json: &str) -> String {
    let mut out = String::with_capacity(json.len() + json.len() / 16);
    let mut in_string = false;
    let mut chars = json.chars().peekable();
    while let Some(ch) = chars.next() {
        if !in_string {
            if ch == '"' {
                in_string = true;
            }
            out.push(ch);
            continue;
        }
        match ch {
            '"' => {
                in_string = false;
                out.push(ch);
            }
            '\\' => match chars.peek() {
                // Valid JSON escape — copy the pair through verbatim.
                Some(&next @ ('"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u')) => {
                    out.push('\\');
                    out.push(next);
                    chars.next();
                }
                // Invalid escape (`\_`, `\(`, …) or a trailing backslash:
                // escape the backslash itself so it becomes a literal.
                _ => out.push_str("\\\\"),
            },
            // Raw control characters are illegal inside a JSON string; escape
            // the common ones and drop anything else unrepresentable.
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {}
            c => out.push(c),
        }
    }
    out
}

/// Extract the first balanced `{ ... }` object from mixed model output.
pub(crate) fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, ch) in text[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start..start + offset + ch.len_utf8()]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Normalize an item name to lowercase snake_case; `None` when empty.
pub(crate) fn normalize_name(raw: &str) -> Option<String> {
    let name: String = raw
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_whitespace() || c == '-' {
                '_'
            } else {
                c
            }
        })
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    let name = name.trim_matches('_').to_string();
    if name.is_empty() {
        return None;
    }
    Some(name.chars().take(MAX_NAME_CHARS).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extraction_schema_declares_the_parsed_fields() {
        let schema = extraction_schema();
        let props = schema["properties"].as_object().unwrap();
        // Every kind array plus `delete` is declared and required.
        for field in ["preferences", "experiences", "skills", "facts", "delete"] {
            assert!(props.contains_key(field), "schema missing '{field}'");
        }
        let required: Vec<&str> = schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(required.contains(&"facts"));
        // Item objects require the fields RawItem reads, including the project flag.
        let item_props = &schema["properties"]["facts"]["items"]["properties"];
        assert!(item_props.get("name").is_some());
        assert!(item_props.get("content").is_some());
        assert!(item_props.get("project_specific").is_some());
    }

    #[test]
    fn schema_conforming_output_parses() {
        // A response shaped exactly as the schema mandates must parse cleanly.
        let response = r#"{
            "preferences": [{"name": "tabs", "content": "prefers tabs", "project_specific": false}],
            "experiences": [], "skills": [],
            "facts": [{"name": "sdk", "content": "uses the vendor SDK", "project_specific": true}],
            "delete": [{"kind": "fact", "name": "old"}]
        }"#;
        let ops = parse_operations(response, Some("svc-a")).unwrap();
        // 2 upserts + 1 delete; the project-specific fact carries the project.
        assert_eq!(ops.len(), 3);
        let assigned = ops.iter().any(
            |op| matches!(op, MemoryOperation::Upsert { project: Some(p), .. } if p == "svc-a"),
        );
        assert!(assigned, "project_specific fact should carry the project");
    }

    #[test]
    fn parses_fenced_json_with_prose() {
        let response = r#"Here are the memories:
```json
{"preferences": [{"name": "Rust Style", "content": "Prefers ? over unwrap"}],
 "experiences": [], "skills": [], "facts": [],
 "delete": [{"kind": "fact", "name": "old_fact"}]}
```"#;
        let ops = parse_operations(response, None).unwrap();
        assert_eq!(ops.len(), 2);
        assert_eq!(
            ops[0],
            MemoryOperation::Upsert {
                kind: MemoryKind::Preference,
                name: "rust_style".to_string(),
                content: "Prefers ? over unwrap".to_string(),
                project: None,
            }
        );
        assert_eq!(
            ops[1],
            MemoryOperation::Delete {
                kind: MemoryKind::Fact,
                name: "old_fact".to_string(),
            }
        );
    }

    #[test]
    fn skips_empty_names_and_content() {
        let response = r#"{"preferences": [{"name": "", "content": "x"},
            {"name": "ok", "content": "  "}], "experiences": [], "skills": [],
            "facts": [], "delete": []}"#;
        let ops = parse_operations(response, None).unwrap();
        assert!(ops.is_empty());
    }

    #[test]
    fn rejects_output_without_json() {
        assert!(parse_operations("I cannot help with that", None).is_err());
    }

    #[test]
    fn extracts_json_with_braces_inside_strings() {
        let response = r#"{"preferences": [{"name": "a", "content": "code: fn x() { y() }"}],
            "experiences": [], "skills": [], "facts": [], "delete": []}"#;
        let ops = parse_operations(response, None).unwrap();
        assert_eq!(ops.len(), 1);
    }

    #[test]
    fn repairs_invalid_markdown_escapes() {
        // `\_` and `\(` are valid Markdown but invalid JSON escapes — the kind
        // of output small local models emit. Strict parsing fails; the repair
        // pass rescues it.
        let response = "{\"facts\": [{\"name\": \"paths\", \"content\": \
            \"use my\\_var and call foo\\(bar\\)\"}], \
            \"preferences\": [], \"experiences\": [], \"skills\": [], \"delete\": []}";
        let ops = parse_operations(response, None).unwrap();
        assert_eq!(ops.len(), 1);
        let MemoryOperation::Upsert { content, .. } = &ops[0] else {
            panic!("expected upsert");
        };
        assert_eq!(content, "use my\\_var and call foo\\(bar\\)");
    }

    #[test]
    fn repairs_lone_backslash_in_string() {
        // A lone backslash followed by a normal char (`\ `, a path separator
        // written raw, …) is an invalid JSON escape the repair pass rescues.
        let response = "{\"facts\": [{\"name\": \"n\", \"content\": \"path C:\\Users\\me\"}], \
            \"preferences\": [], \"experiences\": [], \"skills\": [], \"delete\": []}";
        let ops = parse_operations(response, None).unwrap();
        assert_eq!(ops.len(), 1);
        let MemoryOperation::Upsert { content, .. } = &ops[0] else {
            panic!("expected upsert");
        };
        assert_eq!(content, "path C:\\Users\\me");
    }

    #[test]
    fn repairs_raw_newline_inside_string() {
        // A literal newline inside a string value (no escaping) is invalid JSON.
        let response = "{\"facts\": [{\"name\": \"n\", \"content\": \"line one\nline two\"}], \
            \"preferences\": [], \"experiences\": [], \"skills\": [], \"delete\": []}";
        let ops = parse_operations(response, None).unwrap();
        assert_eq!(ops.len(), 1);
        let MemoryOperation::Upsert { content, .. } = &ops[0] else {
            panic!("expected upsert");
        };
        assert_eq!(content, "line one\nline two");
    }

    #[test]
    fn repair_leaves_valid_escapes_untouched() {
        let valid = r#"{"a": "tab\there \"quoted\" and \\ slash and \n newline"}"#;
        assert_eq!(repair_json_string_escapes(valid), valid);
    }

    #[test]
    fn project_specific_item_is_assigned_to_the_project() {
        let response = r#"{"facts": [
            {"name": "sdk_quirk", "content": "the transport needs a wrapper", "project_specific": true},
            {"name": "prefers_short_fns", "content": "keep functions small", "project_specific": false}
        ], "preferences": [], "experiences": [], "skills": [], "delete": []}"#;
        let ops = parse_operations(response, Some("svc-a")).unwrap();
        let projects: Vec<Option<&str>> = ops
            .iter()
            .map(|op| match op {
                MemoryOperation::Upsert { project, .. } => project.as_deref(),
                _ => None,
            })
            .collect();
        // First is project-specific → carries the project; second is global → None.
        assert_eq!(projects, vec![Some("svc-a"), None]);
    }

    #[test]
    fn project_specific_stays_global_when_project_unknown() {
        // Even when flagged project_specific, an item stays global if the
        // session had no known project — there is nothing to assign it to.
        let response = r#"{"facts": [
            {"name": "sdk_quirk", "content": "x", "project_specific": true}
        ], "preferences": [], "experiences": [], "skills": [], "delete": []}"#;
        let ops = parse_operations(response, None).unwrap();
        let MemoryOperation::Upsert { project, .. } = &ops[0] else {
            panic!("expected upsert");
        };
        assert_eq!(*project, None);
    }

    #[test]
    fn missing_project_specific_defaults_to_global() {
        // Older/looser model output without the field parses as global.
        let response = r#"{"facts": [{"name": "n", "content": "c"}],
            "preferences": [], "experiences": [], "skills": [], "delete": []}"#;
        let ops = parse_operations(response, Some("proj")).unwrap();
        let MemoryOperation::Upsert { project, .. } = &ops[0] else {
            panic!("expected upsert");
        };
        assert_eq!(*project, None);
    }
}
