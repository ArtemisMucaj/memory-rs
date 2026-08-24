//! Memory ingestion — the online write path.
//!
//! Flow (one bounded LLM call with a format-recovery retry):
//!
//! 1. **Idempotence / re-import** — a non-forced re-ingest of a session that
//!    already produced memories is skipped; a forced one hard-deletes the
//!    session's prior memories first.
//! 2. **Prefetch** — semantic-search existing memories for context, so the
//!    model can see what is already on record and phrase new statements
//!    against it instead of duplicating.
//! 3. **Extract** — one call returns atomic facts plus their entity mentions.
//! 4. **Apply** — resolve entity mentions (name-key only), collapse
//!    duplicates, and write.
//!
//! Entity resolution is exact-match on the normalized name key. The
//! embedding-similarity and LLM adjudication tiers the old path used are
//! gone: they were the source of permanent duplicate anchors, and
//! `entity_name_key` normalization catches the variants that matter
//! ("orders-events", "the orders-events service", "orders events" all share
//! one key).

use std::collections::HashMap;
use std::sync::Arc;

use serde::Deserialize;
use tracing::{debug, warn};
use uuid::Uuid;

use openai_rs::ChatClient;

use crate::application::interfaces::{Embedder, MemoryRepository};
use crate::application::use_cases::llm_json::{
    extract_json_object, repair_json_string_escapes, unix_now,
};
use crate::application::use_cases::memory_ingestion_prompt as prompt;
use crate::domain::{
    entity_name_key, DomainError, Entity, Memory, MemoryKind, SessionTranscript, SourceKind,
};

/// How many prior memories are prefetched into the extraction context.
const PREFETCH_LIMIT: usize = 8;

/// Upper bound on memories applied from a single ingestion, guarding
/// against a runaway model flooding the store.
const MAX_MEMORIES_PER_RUN: usize = 32;

/// Default entity type when the model does not supply a usable one.
const UNKNOWN_ENTITY_TYPE: &str = "unknown";

/// Whether two entity types are different enough that the entities must be
/// kept apart.
///
/// Never merge across two entities the model typed differently. `unknown`
/// matches anything, because it asserts nothing.
pub(crate) fn types_conflict(existing_type: &str, new_type: &str) -> bool {
    existing_type != new_type
        && existing_type != UNKNOWN_ENTITY_TYPE
        && new_type != UNKNOWN_ENTITY_TYPE
}

/// What one ingestion produced.
#[derive(Debug, Default, PartialEq)]
pub struct IngestionReport {
    pub memories_written: usize,
    pub entities_created: usize,
    /// Duplicates replaced at write time (same normalized statement text
    /// in the same project).
    pub memories_deduped: usize,
}

/// Outcome of an ingestion request.
#[derive(Debug)]
pub enum IngestionOutcome {
    /// Extraction ran; the report describes what was written.
    Ingested(IngestionReport),
    /// The session already had memories and `force` was not set.
    AlreadyIngested,
}

/// JSON shape the extraction model must return (mirrors [`prompt::schema`]).
#[derive(Debug, Deserialize)]
struct RawIngestion {
    #[serde(default)]
    memories: Vec<RawMemory>,
}

#[derive(Debug, Deserialize)]
struct RawMemory {
    #[serde(default)]
    statement: String,
    #[serde(default)]
    source_kind: String,
    #[serde(default)]
    confidence: f32,
    /// Index of the transcript message the fact came from. Optional — the
    /// model omits it when it cannot place the fact cleanly.
    #[serde(default)]
    source_message_index: Option<i64>,
    #[serde(default)]
    entity_mentions: Vec<RawEntityMention>,
}

#[derive(Debug, Deserialize)]
struct RawEntityMention {
    #[serde(default)]
    name: String,
    #[serde(default, rename = "type")]
    entity_type: String,
}

/// Identity used to collapse duplicate memories within and across sessions:
/// the normalized statement text plus the project scope. Two memories with
/// the same statement about the same project are the same fact.
///
/// Length-prefixed fields rather than a separator — a plain `a|b` encoding
/// lets `(statement="a|b", project="c")` collide with `(statement="a",
/// project="b|c")`, which would silently delete an unrelated memory.
fn duplicate_key(memory: &Memory) -> String {
    let statement = memory.statement.trim().to_lowercase();
    let project = memory.project.as_deref().unwrap_or("");
    format!(
        "{}:{}{}:{}",
        statement.len(),
        statement,
        project.len(),
        project
    )
}

pub struct MemoryIngestionUseCase {
    chat_client: Arc<dyn ChatClient>,
    memory_repo: Arc<dyn MemoryRepository>,
    embedder: Embedder,
}

impl MemoryIngestionUseCase {
    pub fn new(
        chat_client: Arc<dyn ChatClient>,
        memory_repo: Arc<dyn MemoryRepository>,
        embedder: Embedder,
    ) -> Self {
        Self {
            chat_client,
            memory_repo,
            embedder,
        }
    }

    /// Ingest `transcript` into the memory store.
    #[tracing::instrument(skip_all, fields(session_id = %transcript.id))]
    pub async fn execute(
        &self,
        transcript: &SessionTranscript,
        force: bool,
    ) -> Result<IngestionOutcome, DomainError> {
        if force {
            let removed = self
                .memory_repo
                .delete_memories_for_session(&transcript.id)
                .await?;
            if removed > 0 {
                debug!(
                    "memory ingestion: forced re-import removed {removed} prior memories for '{}'",
                    transcript.id
                );
            }
        } else if self
            .memory_repo
            .count_memories_for_session(&transcript.id)
            .await?
            > 0
        {
            return Ok(IngestionOutcome::AlreadyIngested);
        }

        let prior = self.prefetch(transcript).await;
        let raw = self.extract(transcript, &prior).await?;
        let report = self.apply(transcript, raw, &prior).await?;
        Ok(IngestionOutcome::Ingested(report))
    }

    /// Semantic-search existing memories for context. Failures degrade to
    /// "no context" rather than failing the ingest — losing prefetch costs
    /// deduplication quality, but losing the session costs the memory itself.
    async fn prefetch(&self, transcript: &SessionTranscript) -> Vec<Memory> {
        if !self.embedder.embeddings_enabled() {
            return Vec::new();
        }
        let query = prompt::prefetch_query(transcript);
        if query.trim().is_empty() {
            return Vec::new();
        }
        let projects: Vec<String> = transcript.project.clone().into_iter().collect();
        match self.embedder.embed_query(&query).await {
            Ok(vector) => match self
                .memory_repo
                .search_memories_semantic(&vector, Some(&projects), PREFETCH_LIMIT)
                .await
            {
                Ok(results) => results.into_iter().map(|(memory, _)| memory).collect(),
                Err(e) => {
                    warn!("memory prefetch search failed: {e}");
                    Vec::new()
                }
            },
            Err(e) => {
                warn!("memory prefetch embedding failed: {e}");
                Vec::new()
            }
        }
    }

    /// Call the extraction model and parse its JSON, retrying once with a
    /// format-correction message when parsing fails.
    async fn extract(
        &self,
        transcript: &SessionTranscript,
        prior: &[Memory],
    ) -> Result<RawIngestion, DomainError> {
        let system = prompt::system_prompt();
        let user = prompt::user_prompt(transcript, prior);
        let schema = prompt::schema();
        let response = self
            .chat_client
            .complete_json(&system, &user, "memory_ingestion", &schema)
            .await?;
        match parse_ingestion(&response) {
            Ok(parsed) => Ok(parsed),
            Err(first_err) => {
                debug!("memory ingestion output unparseable, retrying once: {first_err}");
                let retry_user = format!("{user}\n\n{}", prompt::format_retry_prompt());
                let response = self
                    .chat_client
                    .complete_json(&system, &retry_user, "memory_ingestion", &schema)
                    .await?;
                parse_ingestion(&response).map_err(|e| {
                    DomainError::parse(format!(
                        "memory ingestion model returned unparseable output twice: {e}"
                    ))
                })
            }
        }
    }

    /// Resolve entity mentions, collapse duplicates, write.
    async fn apply(
        &self,
        transcript: &SessionTranscript,
        raw: RawIngestion,
        prior: &[Memory],
    ) -> Result<IngestionReport, DomainError> {
        let now = unix_now();
        let mut entity_cache: HashMap<String, String> = HashMap::new();
        let mut report = IngestionReport::default();

        // Duplicate index over the prefetched priors, plus everything
        // written in this run so a model that emits the same fact twice
        // writes once.
        let mut seen: HashMap<String, String> = prior
            .iter()
            .map(|c| (duplicate_key(c), c.id.clone()))
            .collect();

        for raw_memory in raw.memories {
            if report.memories_written >= MAX_MEMORIES_PER_RUN {
                debug!(
                    "memory ingestion: hit the {MAX_MEMORIES_PER_RUN}-memory cap, dropping the rest"
                );
                break;
            }
            let statement = raw_memory.statement.trim();
            if statement.is_empty() {
                continue;
            }

            let mut entity_ids = Vec::new();
            for mention in &raw_memory.entity_mentions {
                if let Some(id) = self
                    .resolve_mention(mention, now, &mut entity_cache, &mut report)
                    .await?
                {
                    if !entity_ids.contains(&id) {
                        entity_ids.push(id);
                    }
                }
            }

            let memory = Memory {
                id: Uuid::new_v4().to_string(),
                kind: MemoryKind::Fact,
                statement: statement.to_string(),
                entity_ids,
                project: transcript.project.clone(),
                recorded_at: now,
                source_session_id: Some(transcript.id.clone()),
                source_message_index: raw_memory.source_message_index,
                source_kind: SourceKind::parse(&raw_memory.source_kind)
                    .unwrap_or(SourceKind::Extracted),
                confidence: raw_memory.confidence.clamp(0.0, 1.0),
            };

            let key = duplicate_key(&memory);
            let displaced = seen.get(&key).cloned();

            // Write the new row first. Deleting the displaced memory before
            // the write would leave a window where the store has neither —
            // a failed append would be a lost fact instead of a stale one.
            let vector = self.embed_opt(&memory.statement).await;
            self.memory_repo
                .append_memory(&memory, vector.as_deref())
                .await?;
            report.memories_written += 1;

            if let Some(existing_id) = displaced {
                self.memory_repo.delete_memory(&existing_id).await?;
                report.memories_deduped += 1;
            }
            seen.insert(key, memory.id.clone());
        }
        Ok(report)
    }

    /// Resolve one entity mention to an entity id, creating the entity on
    /// first sight.
    ///
    /// The only tier is exact match on the normalized name key. The
    /// embedding and adjudication tiers the old path used were the source
    /// of permanent duplicate anchors — `entity_name_key` normalization
    /// now catches the variants that matter, and a miss simply creates a
    /// new entity, which is the recoverable failure.
    async fn resolve_mention(
        &self,
        mention: &RawEntityMention,
        now: i64,
        cache: &mut HashMap<String, String>,
        report: &mut IngestionReport,
    ) -> Result<Option<String>, DomainError> {
        let trimmed = mention.name.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        // Anything outside the closed vocabulary is remapped to `unknown` —
        // a malformed response such as `database` must not become a
        // permanent unsupported type that blocks later valid mentions from
        // merging with it.
        let entity_type = match mention.entity_type.trim().to_lowercase().as_str() {
            "" => UNKNOWN_ENTITY_TYPE.to_string(),
            t if crate::domain::VALID_ENTITY_TYPES.contains(&t) => t.to_string(),
            _ => UNKNOWN_ENTITY_TYPE.to_string(),
        };
        let key = format!("{}\u{0}{entity_type}", entity_name_key(trimmed));
        if let Some(id) = cache.get(&key) {
            return Ok(Some(id.clone()));
        }
        // The key is broader than the name, so a hit can be a differently-
        // typed entity that merely shares it. Take the first the type guard
        // accepts.
        let named = self.memory_repo.find_entities_by_name(trimmed).await?;
        if let Some(existing) = named
            .into_iter()
            .find(|e| !types_conflict(&e.entity_type, &entity_type))
        {
            cache.insert(key, existing.id.clone());
            return Ok(Some(existing.id));
        }

        let entity = Entity {
            id: Uuid::new_v4().to_string(),
            entity_type,
            canonical_name: trimmed.to_string(),
            names: Vec::new(),
            created_at: now,
            updated_at: now,
        };
        self.memory_repo.upsert_entity(&entity).await?;
        report.entities_created += 1;
        cache.insert(key, entity.id.clone());
        Ok(Some(entity.id))
    }

    /// Embed `text`, returning `None` when embeddings are disabled or the
    /// call fails (the row stays keyword-searchable either way).
    async fn embed_opt(&self, text: &str) -> Option<Vec<f32>> {
        if !self.embedder.embeddings_enabled() {
            return None;
        }
        match self.embedder.embed_query(text).await {
            Ok(vector) => Some(vector),
            Err(e) => {
                warn!("failed to embed memory text: {e}");
                None
            }
        }
    }
}

/// Parse the model's ingestion JSON, tolerating prose/fences and the
/// invalid-escape output small local models emit.
fn parse_ingestion(response: &str) -> Result<RawIngestion, DomainError> {
    let json = extract_json_object(response)
        .ok_or_else(|| DomainError::parse("no JSON object found in ingestion output"))?;
    match serde_json::from_str::<RawIngestion>(json) {
        Ok(parsed) => Ok(parsed),
        Err(strict_err) => {
            let repaired = repair_json_string_escapes(json);
            serde_json::from_str::<RawIngestion>(&repaired)
                .map_err(|_| DomainError::parse(format!("invalid ingestion JSON: {strict_err}")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory_with(id: &str, source_kind: SourceKind, confidence: f32) -> Memory {
        Memory {
            id: id.to_string(),
            kind: MemoryKind::Fact,
            statement: "the team uses svc-a".to_string(),
            entity_ids: vec!["entity-1".to_string()],
            project: Some("owner/repo".to_string()),
            recorded_at: 100,
            source_session_id: Some("session-1".to_string()),
            source_message_index: None,
            source_kind,
            confidence,
        }
    }

    /// A type conflict keeps entities apart. Merging two differently-typed
    /// entities cannot be undone — there is no supersession chain for an
    /// entity — so even a perfect name match must not do it.
    #[test]
    fn a_type_conflict_keeps_entities_apart() {
        assert!(types_conflict("service", "tool"));
        assert!(types_conflict("person", "library"));
        assert!(!types_conflict("tool", "tool"));
    }

    /// `unknown` asserts nothing, so it must not block a match in either
    /// direction — otherwise the model's uncertainty would permanently
    /// fragment the entity set.
    #[test]
    fn unknown_type_matches_anything() {
        assert!(!types_conflict(UNKNOWN_ENTITY_TYPE, "tool"));
        assert!(!types_conflict("tool", UNKNOWN_ENTITY_TYPE));
    }

    #[test]
    fn duplicate_key_ignores_statement_case() {
        let mut a = memory_with("a", SourceKind::UserStated, 0.5);
        let mut b = memory_with("b", SourceKind::Extracted, 0.9);
        a.statement = "The team uses SVC-A.".to_string();
        b.statement = "the team uses svc-a.".to_string();
        assert_eq!(duplicate_key(&a), duplicate_key(&b));
    }

    #[test]
    fn duplicate_key_separates_scopes() {
        let base = memory_with("a", SourceKind::UserStated, 0.5);
        let mut other_project = base.clone();
        other_project.project = Some("owner/other".to_string());
        assert_ne!(duplicate_key(&base), duplicate_key(&other_project));

        let mut global = base.clone();
        global.project = None;
        assert_ne!(duplicate_key(&base), duplicate_key(&global));
    }

    /// The encoding must not let a separator inside one field spill into the
    /// next — `(statement="a|b", project="c")` and `(statement="a",
    /// project="b|c")` would otherwise produce the same key, and the dedupe
    /// pass would delete an unrelated memory.
    #[test]
    fn duplicate_key_is_not_ambiguous_across_field_boundaries() {
        let mut a = memory_with("a", SourceKind::UserStated, 0.5);
        a.statement = "a|b".to_string();
        a.project = Some("c".to_string());

        let mut b = memory_with("b", SourceKind::UserStated, 0.5);
        b.statement = "a".to_string();
        b.project = Some("b|c".to_string());

        assert_ne!(duplicate_key(&a), duplicate_key(&b));
    }

    #[test]
    fn parses_a_memory_with_entity_mentions() {
        let response = r#"{"memories": [
            {"statement": "the user prefers tabs",
             "source_kind": "user_stated", "confidence": 0.9,
             "entity_mentions": [{"name": "the user", "type": "person"}]}
        ]}"#;
        let parsed = parse_ingestion(response).unwrap();
        assert_eq!(parsed.memories.len(), 1);
        let c = &parsed.memories[0];
        assert_eq!(c.statement, "the user prefers tabs");
        assert_eq!(c.entity_mentions.len(), 1);
        assert_eq!(c.entity_mentions[0].name, "the user");
        assert_eq!(c.entity_mentions[0].entity_type, "person");
    }

    #[test]
    fn parses_fenced_json() {
        let response = "Sure!\n```json\n{\"memories\": [\
            {\"statement\": \"prefers tabs\", \
             \"source_kind\": \"user_stated\", \"confidence\": 0.8, \
             \"entity_mentions\": []}]}\n```\nDone.";
        let parsed = parse_ingestion(response).unwrap();
        assert_eq!(parsed.memories.len(), 1);
    }

    /// Small local models emit markdown escapes that are illegal JSON. The
    /// repair pass is what keeps one stray backslash from costing a whole
    /// session's memory.
    #[test]
    fn repairs_invalid_escapes_before_giving_up() {
        let response = r#"{"memories": [{"statement": "the user prefers snake\_case names",
            "source_kind": "user_stated", "confidence": 0.7, "entity_mentions": []}]}"#;
        let parsed = parse_ingestion(response).unwrap();
        assert_eq!(parsed.memories.len(), 1);
        assert!(parsed.memories[0].statement.contains("snake"));
    }

    #[test]
    fn unknown_source_kind_falls_back_rather_than_dropping_the_memory() {
        assert_eq!(
            SourceKind::parse("nonsense").unwrap_or(SourceKind::Extracted),
            SourceKind::Extracted,
        );
    }

    #[test]
    fn rejects_output_without_json() {
        assert!(parse_ingestion("I could not extract anything").is_err());
    }

    #[test]
    fn empty_memory_list_parses() {
        assert!(parse_ingestion(r#"{"memories": []}"#)
            .unwrap()
            .memories
            .is_empty());
    }
}
