//! Virtual-filesystem summarization — the L0/L1 half of the memory system.
//!
//! Where [`memory_extraction`](super::memory_extraction) distills a session
//! into flat [`MemoryItem`]s, this use case builds the *navigable* layer over
//! them: nodes carrying an L0 abstract and an L1 overview so an agent reads a
//! summary first and drills into detail only when needed.
//!
//! Four things get summarized:
//!
//! 1. **Each imported session** → `memory://sessions/<id>` with its full
//!    normalized transcript as L2, plus a generated abstract + overview.
//! 2. **Each explicitly-added resource** → `memory://resources/<slug>` with the
//!    fetched file/page text as L2, plus a generated abstract + overview.
//! 3. **The whole memory store** → the `memory://memory` digest: a regenerated
//!    abstract + overview over every stored item, meant to be read first.
//! 4. **Each project/namespace project** → `memory://projects/<project>`: a digest
//!    over the items carrying that project, read first when working in that
//!    project. Regenerated lazily — only when the project's items changed.
//!
//! Each uses one small LLM call (the same [`ChatClient`] extraction uses), with
//! a single format-recovery retry and a deterministic fallback so a flaky model
//! never blocks the operation.

use std::sync::Arc;

use openai_rs::ChatClient;
use serde::Deserialize;
use tracing::{debug, warn};

use crate::application::interfaces::{Embedder, MemoryRepository, NodeRepository};
use crate::application::use_cases::llm_json::{extract_json_object, unix_now};
use crate::domain::{
    DomainError, Memory, MemoryNode, MemoryStatus, NodeKind, SessionMessage, SessionTranscript,
};

/// Root URI of the memory digest node ("read this first").
pub const MEMORY_ROOT_URI: &str = "memory://memory";
/// Parent directory URI under which per-project/namespace project digests live.
pub const PROJECTS_ROOT_URI: &str = "memory://projects";
/// Parent directory URI under which per-session nodes live.
pub const SESSIONS_ROOT_URI: &str = "memory://sessions";
/// Parent directory URI under which explicitly-added resources (files/URLs)
/// live.
pub const RESOURCES_ROOT_URI: &str = "memory://resources";

/// Maximum characters of transcript sent to the summarization model. The full
/// transcript is still *stored* as L2; only the summarization prompt is capped.
const MAX_SUMMARY_INPUT_CHARS: usize = 40_000;

/// Maximum characters of a resource's extracted text kept as L2. Web pages and
/// large files are truncated here so a single node cannot bloat the store; the
/// truncation is flagged in the stored content.
const MAX_RESOURCE_CONTENT_CHARS: usize = 200_000;

/// Maximum characters of a single abstract (L0) kept after generation.
const MAX_ABSTRACT_CHARS: usize = 400;
/// Maximum characters of a single overview (L1) kept after generation.
const MAX_OVERVIEW_CHARS: usize = 2_000;

/// Most memories fed into a digest prompt.
///
/// Memories are roughly an order of magnitude more numerous than the items they
/// replaced — one per fact rather than one per topic — so a whole-store digest
/// would otherwise be truncated mid-list by `MAX_SUMMARY_INPUT_CHARS`. That
/// truncation is what makes the cap matter: cutting a *character* budget takes
/// whatever happened to be listed first, so the memories must be **ranked** and
/// cut here, deliberately, before the character clamp gets to them.
const MAX_DIGEST_MEMORIES: usize = 400;

/// Builds and maintains the memory virtual filesystem's L0/L1 nodes.
pub struct SummarizeMemoryUseCase {
    chat_client: Arc<dyn ChatClient>,
    node_repo: Arc<dyn NodeRepository>,
    /// Digests summarize memories; sessions and resources are still nodes, so
    /// both ports are needed.
    memory_repo: Arc<dyn MemoryRepository>,
    embedding_service: Arc<Embedder>,
}

impl SummarizeMemoryUseCase {
    pub fn new(
        chat_client: Arc<dyn ChatClient>,
        node_repo: Arc<dyn NodeRepository>,
        memory_repo: Arc<dyn MemoryRepository>,
        embedding_service: Arc<Embedder>,
    ) -> Self {
        Self {
            chat_client,
            node_repo,
            memory_repo,
            embedding_service,
        }
    }

    /// Active memories, ranked and capped for a digest prompt.
    ///
    /// Ranked by confidence then recency so the cap keeps the memories most worth
    /// indexing, rather than an arbitrary slice of whatever the store returned.
    async fn digest_memories(&self) -> Result<Vec<Memory>, DomainError> {
        let mut memories = self
            .memory_repo
            .list_memories(None, Some(MemoryStatus::Active), None)
            .await?;
        memories.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.recorded_at.cmp(&a.recorded_at))
                .then_with(|| a.id.cmp(&b.id))
        });
        memories.truncate(MAX_DIGEST_MEMORIES);
        Ok(memories)
    }

    /// Store `transcript` as a session node (`memory://sessions/<id>`) with a
    /// generated L0 abstract + L1 overview and its full transcript as L2.
    ///
    /// Summarization is best-effort: on model/embedding failure the node is
    /// still written with a deterministic fallback summary so the transcript
    /// is never lost.
    #[tracing::instrument(skip_all, fields(session_id = %transcript.id))]
    pub async fn summarize_session(
        &self,
        transcript: &SessionTranscript,
    ) -> Result<MemoryNode, DomainError> {
        let content = render_transcript(&transcript.messages);
        let (abstract_, overview) = match self
            .generate(
                &session_system_prompt(),
                &session_user_prompt(transcript, &content),
            )
            .await
        {
            Some(summary) => summary,
            None => fallback_session_summary(transcript),
        };

        let uri = format!("{SESSIONS_ROOT_URI}/{}", transcript.id);
        let now = unix_now();
        let created_at = match self.node_repo.find_node(&uri).await {
            Ok(Some(prev)) => prev.created_at(),
            _ => now,
        };
        let node = MemoryNode::new(
            uri,
            NodeKind::Session,
            Some(SESSIONS_ROOT_URI.to_string()),
            clamp(&abstract_, MAX_ABSTRACT_CHARS),
            clamp(&overview, MAX_OVERVIEW_CHARS),
            content,
            created_at,
            now,
        );
        let vector = self.embed_node(&node).await;
        self.node_repo.upsert_node(&node, vector.as_deref()).await?;
        Ok(node)
    }

    /// Store an explicitly-added resource as a node
    /// (`memory://resources/<slug>`) with the fetched `text` as its L2 detail
    /// and a generated L0 abstract + L1 overview.
    ///
    /// `slug` is the snake_case identifier for the node (unique per resource);
    /// `source` is the original URL or file path, recorded for provenance.
    /// Best-effort like the other summaries: on model failure a deterministic
    /// fallback is used so the resource is still stored.
    #[tracing::instrument(skip_all, fields(resource = %source))]
    pub async fn summarize_resource(
        &self,
        slug: &str,
        source: &str,
        text: &str,
    ) -> Result<MemoryNode, DomainError> {
        let content = clamp_with_marker(text, MAX_RESOURCE_CONTENT_CHARS);
        let (abstract_, overview) = match self
            .generate(
                &resource_system_prompt(),
                &resource_user_prompt(source, &content),
            )
            .await
        {
            Some(summary) => summary,
            None => fallback_resource_summary(source, &content),
        };

        let uri = format!("{RESOURCES_ROOT_URI}/{slug}");
        let now = unix_now();
        let created_at = match self.node_repo.find_node(&uri).await {
            Ok(Some(prev)) => prev.created_at(),
            _ => now,
        };
        let node = MemoryNode::new(
            uri,
            NodeKind::Resource,
            Some(RESOURCES_ROOT_URI.to_string()),
            clamp(&abstract_, MAX_ABSTRACT_CHARS),
            clamp(&overview, MAX_OVERVIEW_CHARS),
            content,
            created_at,
            now,
        );
        let vector = self.embed_node(&node).await;
        self.node_repo.upsert_node(&node, vector.as_deref()).await?;
        Ok(node)
    }

    /// Regenerate the whole-memory digest (`memory://memory`) from the current
    /// set of stored items: a fresh L0 abstract + L1 overview read before
    /// drilling into individual memories.
    ///
    /// With zero or one item there is nothing to summarize, so a deterministic
    /// placeholder is written without spending an LLM call.
    #[tracing::instrument(skip_all)]
    pub async fn regenerate_digest(&self) -> Result<MemoryNode, DomainError> {
        let memories = self.digest_memories().await?;
        let (abstract_, overview) = if memories.len() < 2 {
            fallback_digest_summary(&memories)
        } else {
            match self
                .generate(&digest_system_prompt(), &digest_user_prompt(&memories))
                .await
            {
                Some(summary) => summary,
                None => fallback_digest_summary(&memories),
            }
        };

        let now = unix_now();
        let created_at = match self.node_repo.find_node(MEMORY_ROOT_URI).await {
            Ok(Some(prev)) => prev.created_at(),
            _ => now,
        };
        let node = MemoryNode::new(
            MEMORY_ROOT_URI.to_string(),
            NodeKind::Memory,
            None,
            clamp(&abstract_, MAX_ABSTRACT_CHARS),
            clamp(&overview, MAX_OVERVIEW_CHARS),
            String::new(),
            created_at,
            now,
        );
        let vector = self.embed_node(&node).await;
        self.node_repo.upsert_node(&node, vector.as_deref()).await?;
        Ok(node)
    }

    /// Regenerate the per-project digest nodes (`memory://projects/<project>`),
    /// one per distinct project/namespace project found on stored items: the
    /// index an agent reads first when working in that project.
    ///
    /// Cheap to call repeatedly: a project's digest is only regenerated when one
    /// of its items changed since the node was last written, and digests whose
    /// project no longer exists (all items deleted or promoted to global) are
    /// removed. Returns how many digests were (re)generated.
    #[tracing::instrument(skip_all)]
    pub async fn regenerate_project_digests(&self) -> Result<usize, DomainError> {
        let memories = self.digest_memories().await?;
        let mut by_project: std::collections::BTreeMap<&str, Vec<&Memory>> =
            std::collections::BTreeMap::new();
        for memory in &memories {
            if let Some(project) = memory.project.as_deref() {
                by_project.entry(project).or_default().push(memory);
            }
        }

        // Drop digests for projects that vanished from the store.
        let existing = self.node_repo.list_nodes(Some(NodeKind::Project)).await?;
        let live_uris: std::collections::HashSet<String> = by_project
            .keys()
            .map(|project| project_digest_uri(project))
            .collect();
        for node in &existing {
            if !live_uris.contains(node.uri()) {
                if let Err(e) = self.node_repo.delete_node(node.uri()).await {
                    warn!(
                        "failed to delete stale project digest '{}': {e}",
                        node.uri()
                    );
                }
            }
        }

        let mut regenerated = 0usize;
        for (project, project_memories) in by_project {
            let uri = project_digest_uri(project);
            let previous = self.node_repo.find_node(&uri).await?;
            // A sorted concatenation of item IDs + timestamps: it changes on any
            // content edit, deletion, move, or addition, and is stored as the
            // node's content so the next run can detect a change. Computed once
            // and reused for both the staleness check and the written node.
            // Memories are immutable, so a memory's id fully determines its
            // content — the manifest needs no timestamp to detect an edit,
            // because an "edit" is a different memory with a different id. That
            // makes this staleness check strictly more reliable than the item
            // version it replaces, which had to guess from `updated_at`.
            let manifest: String = {
                let mut ids: Vec<&str> = project_memories.iter().map(|c| c.id.as_str()).collect();
                ids.sort_unstable();
                ids.join(";")
            };
            if previous
                .as_ref()
                .is_some_and(|prev| prev.content() == manifest)
            {
                continue;
            }

            let (abstract_, overview) = if project_memories.len() < 2 {
                fallback_project_digest_summary(project, &project_memories)
            } else {
                match self
                    .generate(
                        &project_digest_system_prompt(),
                        &project_digest_user_prompt(project, &project_memories),
                    )
                    .await
                {
                    Some(summary) => summary,
                    None => fallback_project_digest_summary(project, &project_memories),
                }
            };

            let now = unix_now();
            let created_at = previous.map(|p| p.created_at()).unwrap_or(now);
            let node = MemoryNode::new(
                uri,
                NodeKind::Project,
                Some(PROJECTS_ROOT_URI.to_string()),
                clamp(&abstract_, MAX_ABSTRACT_CHARS),
                clamp(&overview, MAX_OVERVIEW_CHARS),
                manifest,
                created_at,
                now,
            )
            // Carry the original project string (git remote / namespace) as the
            // display label — the URI slugifies it lossily, so this is what a
            // client should show instead of `github_com_org_repo-<hash>`.
            .with_label(project);
            let vector = self.embed_node(&node).await;
            self.node_repo.upsert_node(&node, vector.as_deref()).await?;
            regenerated += 1;
        }
        Ok(regenerated)
    }

    /// Run one summarization call, parsing `{abstract, overview}` JSON with a
    /// single format-recovery retry. Returns `None` on any failure so callers
    /// fall back to a deterministic summary instead of aborting the import.
    async fn generate(&self, system: &str, user: &str) -> Option<(String, String)> {
        match self.chat_client.complete(system, user).await {
            Ok(response) => match parse_summary(&response) {
                Some(summary) => return Some(summary),
                None => debug!("summary output unparseable, retrying once"),
            },
            Err(e) => {
                warn!("summary generation failed: {e}");
                return None;
            }
        }
        let retry_user = format!("{user}\n\n{}", SUMMARY_RETRY_PROMPT);
        match self.chat_client.complete(system, &retry_user).await {
            Ok(response) => parse_summary(&response),
            Err(e) => {
                warn!("summary generation retry failed: {e}");
                None
            }
        }
    }

    /// Embed the node's L0/L1 summary for semantic recall; `None` when
    /// embeddings are disabled or fail (the node stays keyword-searchable).
    async fn embed_node(&self, node: &MemoryNode) -> Option<Vec<f32>> {
        if !self.embedding_service.embeddings_enabled() {
            return None;
        }
        match self
            .embedding_service
            .embed_query(&node.embedding_text())
            .await
        {
            Ok(vector) => Some(vector),
            Err(e) => {
                warn!("failed to embed memory node '{}': {e}", node.uri());
                None
            }
        }
    }
}

/// Render a transcript to a stored L2 body: `[idx][role]: content` lines,
/// full (not elided — this is the archived detail, not a prompt).
fn render_transcript(messages: &[SessionMessage]) -> String {
    messages
        .iter()
        .enumerate()
        .filter(|(_, m)| !m.content.trim().is_empty())
        .map(|(idx, m)| format!("[{}][{}]: {}", idx, m.role, m.content.trim()))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// JSON shape both summarization prompts must return.
#[derive(Debug, Deserialize)]
struct SummaryOutput {
    #[serde(default)]
    r#abstract: String,
    #[serde(default)]
    overview: String,
}

const SUMMARY_RETRY_PROMPT: &str =
    "Your previous output could not be parsed. Output ONLY a JSON object with exactly two \
     string fields, \"abstract\" and \"overview\". No prose, no markdown fence.";

fn session_system_prompt() -> String {
    r#"You summarize a finished coding-assistant session for a two-level index.
Produce:
- "abstract": ONE sentence (max ~30 words) capturing what the session was about and its outcome — this is what a reader scans first to decide whether to open the session.
- "overview": 3-5 markdown bullet points covering the arc of the session — the goal, the key steps/decisions, and the result. No preamble.

Focus on durable substance (what was done, decided, or learned), not conversational filler.

Output ONLY a JSON object: {"abstract": "...", "overview": "..."}"#
        .to_string()
}

fn session_user_prompt(transcript: &SessionTranscript, rendered: &str) -> String {
    let mut prompt = String::new();
    if let (Some(start), Some(end)) = (transcript.started_at(), transcript.ended_at()) {
        if start == end {
            prompt.push_str(&format!("Session time: {start}\n\n"));
        } else {
            prompt.push_str(&format!("Session time: {start} - {end}\n\n"));
        }
    }
    prompt.push_str("## Transcript\n\n");
    prompt.push_str(&clamp(rendered, MAX_SUMMARY_INPUT_CHARS));
    prompt.push_str("\n\nSummarize this session as the specified JSON object.");
    prompt
}

fn resource_system_prompt() -> String {
    r#"You summarize a document or web page that a user has added to their knowledge base, for a two-level index.
Produce:
- "abstract": ONE sentence (max ~30 words) capturing what the resource is and what it covers — what a reader scans first to decide whether to open it.
- "overview": 3-6 markdown bullet points covering the resource's main topics, structure, or key takeaways, so the reader knows what is inside and whether to drill into the full text.

Summarize only what the content actually says; do not invent. Output ONLY a JSON object: {"abstract": "...", "overview": "..."}"#
        .to_string()
}

fn resource_user_prompt(source: &str, content: &str) -> String {
    let mut prompt = format!("Source: {source}\n\n## Content\n\n");
    // Large resources: keep the head and tail so the summary reflects the whole.
    prompt.push_str(&head_tail(content, MAX_SUMMARY_INPUT_CHARS));
    prompt.push_str("\n\nSummarize this resource as the specified JSON object.");
    prompt
}

fn digest_system_prompt() -> String {
    r#"You maintain a top-level index of an assistant's long-term memory about a user and their project.
You are given the stored memories (preferences, experiences, skills, facts) as atomic statements.
Produce a summary an agent reads FIRST, before drilling into individual memories:
- "abstract": ONE sentence (max ~35 words) capturing who this user is and what the memory covers at a glance.
- "overview": a markdown outline grouping what is known by theme (e.g. preferences, project facts, reusable experiences), naming the notable items so the reader knows what exists and can drill in. Keep it scannable.

Do not invent anything not present in the memories. Output ONLY a JSON object: {"abstract": "...", "overview": "..."}"#
        .to_string()
}

fn digest_user_prompt(memories: &[Memory]) -> String {
    const MAX_MEMORY_CHARS: usize = 400;
    let mut prompt = String::from("## Stored memories\n\n");
    for memory in memories {
        prompt.push_str(&format!(
            "- [{}] {}\n",
            memory.kind.as_str(),
            clamp(&one_line(&memory.statement), MAX_MEMORY_CHARS)
        ));
    }
    prompt.push_str("\n\nSummarize the memory store as the specified JSON object.");
    clamp(&prompt, MAX_SUMMARY_INPUT_CHARS)
}

/// URI of the digest node for one project/namespace.
///
/// `resource_slug` is lossy — `docs/api.v2` and `docs_api_v2` slug identically —
/// so a short hash of the *original* project string is appended to keep distinct
/// projects on distinct URIs (otherwise their digests would overwrite each other
/// and stale-node cleanup could not tell them apart). The readable slug is kept
/// as a human-friendly prefix.
fn project_digest_uri(project: &str) -> String {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(project.as_bytes());
    let short: String = hash.iter().take(4).map(|b| format!("{b:02x}")).collect();
    format!("{PROJECTS_ROOT_URI}/{}-{short}", resource_slug(project))
}

fn project_digest_system_prompt() -> String {
    r#"You maintain the index of an assistant's long-term memory about ONE project (or one namespace of related projects).
You are given the memory items belonging to that project (preferences, experiences, skills, facts).
Produce a summary an agent working in this project reads FIRST, before drilling into individual memories:
- "abstract": ONE sentence (max ~35 words) capturing what this project is and what the memory covers at a glance.
- "overview": a markdown outline grouping what is known by theme, naming the notable items so the reader knows what exists and can drill in. Keep it scannable.

Do not invent anything not present in the memories. Output ONLY a JSON object: {"abstract": "...", "overview": "..."}"#
        .to_string()
}

fn project_digest_user_prompt(project: &str, memories: &[&Memory]) -> String {
    const MAX_MEMORY_CHARS: usize = 400;
    let mut prompt = format!("## Memories belonging to project '{project}'\n\n");
    for memory in memories {
        prompt.push_str(&format!(
            "- [{}] {}\n",
            memory.kind.as_str(),
            clamp(&one_line(&memory.statement), MAX_MEMORY_CHARS)
        ));
    }
    prompt.push_str("\n\nSummarize this project's memory as the specified JSON object.");
    clamp(&prompt, MAX_SUMMARY_INPUT_CHARS)
}

/// Deterministic fallback for a project digest when there is little to summarize
/// or the model is unavailable.
fn fallback_project_digest_summary(project: &str, memories: &[&Memory]) -> (String, String) {
    let mut overview = format!("Memories belonging to '{project}':\n");
    for memory in memories {
        overview.push_str(&format!(
            "- [{}] {}\n",
            memory.kind.as_str(),
            one_line(&memory.statement)
        ));
    }
    (
        format!(
            "{} stored memories about project '{project}'.",
            memories.len()
        ),
        overview,
    )
}

/// Deterministic fallback used when a session cannot be summarized by the model.
fn fallback_session_summary(transcript: &SessionTranscript) -> (String, String) {
    let msg_count = transcript
        .messages
        .iter()
        .filter(|m| !m.content.trim().is_empty())
        .count();
    let first_user = transcript
        .messages
        .iter()
        .find(|m| m.role == "user" && !m.content.trim().is_empty())
        .map(|m| clamp(&one_line(&m.content), 200))
        .unwrap_or_else(|| "(no user message)".to_string());
    (
        format!(
            "Imported session '{}' ({msg_count} messages).",
            transcript.id
        ),
        format!(
            "- Session id: {}\n- Messages: {msg_count}\n- Opened with: {first_user}",
            transcript.id
        ),
    )
}

/// Deterministic fallback for the digest when there is nothing to summarize or
/// the model is unavailable.
fn fallback_digest_summary(memories: &[Memory]) -> (String, String) {
    if memories.is_empty() {
        return (
            "No memories stored yet.".to_string(),
            "- The memory store is empty. Import a session to populate it.".to_string(),
        );
    }
    let mut overview = String::from("Stored memories:\n");
    for memory in memories {
        overview.push_str(&format!(
            "- [{}] {}\n",
            memory.kind.as_str(),
            one_line(&memory.statement)
        ));
    }
    (
        format!(
            "{} stored memories about the user and project.",
            memories.len()
        ),
        overview,
    )
}

/// Deterministic fallback for a resource when the model is unavailable: use the
/// source and the first non-empty line of the content.
fn fallback_resource_summary(source: &str, content: &str) -> (String, String) {
    let first_line = content
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .map(|l| clamp(&one_line(l), 200))
        .unwrap_or_else(|| "(empty)".to_string());
    (
        format!("Resource added from {source}."),
        format!("- Source: {source}\n- Starts with: {first_line}"),
    )
}

/// Parse the model's `{abstract, overview}` response, tolerating prose or a
/// markdown fence around the object. `None` when no usable object is found.
fn parse_summary(response: &str) -> Option<(String, String)> {
    let json = extract_json_object(response)?;
    let output: SummaryOutput = serde_json::from_str(json).ok()?;
    let abstract_ = output.r#abstract.trim().to_string();
    let overview = output.overview.trim().to_string();
    if abstract_.is_empty() && overview.is_empty() {
        return None;
    }
    Some((abstract_, overview))
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Normalize a title or user-supplied name into a lowercase snake_case slug
/// suitable for a resource node URI. Returns a stable fallback when the input
/// reduces to nothing.
pub fn resource_slug(raw: &str) -> String {
    const MAX_SLUG_CHARS: usize = 64;
    let slug: String = raw
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_whitespace() || c == '-' || c == '.' || c == '/' {
                '_'
            } else {
                c
            }
        })
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    let slug = slug
        .split('_')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("_");
    if slug.is_empty() {
        return "resource".to_string();
    }
    slug.chars().take(MAX_SLUG_CHARS).collect()
}

fn clamp(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let truncated: String = text.chars().take(max_chars.saturating_sub(3)).collect();
    format!("{truncated}...")
}

/// Like [`clamp`] but appends an explicit truncation marker, for stored L2
/// content where the reader should know the tail was dropped.
fn clamp_with_marker(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let kept: String = text.chars().take(max_chars).collect();
    format!("{kept}\n\n[... resource truncated at {max_chars} characters ...]")
}

/// Keep the head and tail of `text` within a char budget, eliding the middle —
/// so a summary of a large resource reflects both its start and its end.
fn head_tail(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let head_budget = max_chars / 2;
    let tail_budget = max_chars - head_budget;
    let head: String = text.chars().take(head_budget).collect();
    let tail: String = {
        let all: Vec<char> = text.chars().collect();
        all[all.len() - tail_budget..].iter().collect()
    };
    format!("{head}\n\n[... middle elided ...]\n\n{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fenced_summary() {
        let response = r#"Here you go:
```json
{"abstract": "Fixed a flaky test", "overview": "- Found the race\n- Added a lock"}
```"#;
        let (a, o) = parse_summary(response).unwrap();
        assert_eq!(a, "Fixed a flaky test");
        assert!(o.contains("Found the race"));
    }

    #[test]
    fn rejects_summary_without_json() {
        assert!(parse_summary("I cannot help with that").is_none());
    }

    #[test]
    fn rejects_empty_summary() {
        assert!(parse_summary(r#"{"abstract": "", "overview": ""}"#).is_none());
    }

    #[test]
    fn renders_transcript_skipping_empty() {
        let messages = vec![
            SessionMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
                timestamp: None,
            },
            SessionMessage {
                role: "assistant".to_string(),
                content: "  ".to_string(),
                timestamp: None,
            },
            SessionMessage {
                role: "assistant".to_string(),
                content: "hi".to_string(),
                timestamp: None,
            },
        ];
        let rendered = render_transcript(&messages);
        assert!(rendered.contains("[0][user]: hello"));
        assert!(rendered.contains("[2][assistant]: hi"));
        assert!(!rendered.contains("[1]"));
    }

    #[test]
    fn empty_store_digest_fallback() {
        let (a, o) = fallback_digest_summary(&[]);
        assert!(a.contains("No memories"));
        assert!(o.contains("empty"));
    }

    #[test]
    fn resource_slug_normalizes_titles() {
        assert_eq!(resource_slug("My Cool Guide!"), "my_cool_guide");
        assert_eq!(resource_slug("docs/api.v2"), "docs_api_v2");
        assert_eq!(resource_slug("  --- "), "resource");
        assert_eq!(resource_slug("Already_Snake"), "already_snake");
    }

    #[test]
    fn clamp_with_marker_flags_truncation() {
        let short = clamp_with_marker("abc", 100);
        assert_eq!(short, "abc");
        let long = clamp_with_marker(&"x".repeat(50), 10);
        assert!(long.contains("resource truncated"));
    }

    #[test]
    fn head_tail_keeps_both_ends() {
        let text: String = ('a'..='z').collect();
        let ht = head_tail(&text, 10);
        assert!(ht.starts_with("abcde"));
        assert!(ht.trim_end().ends_with("vwxyz"));
        assert!(ht.contains("elided"));
    }

    #[test]
    fn resource_fallback_uses_source_and_first_line() {
        let (a, o) = fallback_resource_summary("https://x.dev/p", "\n\n  First real line\nmore");
        assert!(a.contains("https://x.dev/p"));
        assert!(o.contains("First real line"));
    }
}
