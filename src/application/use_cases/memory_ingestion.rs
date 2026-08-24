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
//! 3. **Extract** — one call returns atomic subject–predicate–object facts.
//! 4. **Apply** — resolve entities (name-key only), collapse duplicates, and
//!    write.
//!
//! There is no edge writing, no corroboration, no supersession. A duplicate
//! detected at write time replaces the prior row outright (hard delete +
//! insert), and entity resolution is exact-match on the normalized name key
//! — the embedding and LLM adjudication tiers were the source of permanent
//! duplicate anchors and are gone.

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
    entity_name_key, DomainError, Entity, EntityRef, Memory, MemoryKind, Predicate,
    SessionTranscript, SourceKind,
};

/// How many prior memories are prefetched into the extraction context.
const PREFETCH_LIMIT: usize = 8;

/// Upper bound on memories applied from a single ingestion, guarding against
/// a runaway model flooding the store.
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
    /// Duplicates replaced at write time (same subject/predicate/object key).
    pub memories_deduped: usize,
    /// Memories whose predicate fell outside the closed vocabulary and landed
    /// on `relates_to`. The metric for whether [`Predicate`] needs extending.
    pub predicates_out_of_vocabulary: usize,
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
    subject: String,
    #[serde(default)]
    subject_is_entity: bool,
    #[serde(default)]
    subject_type: String,
    #[serde(default)]
    predicate: String,
    #[serde(default)]
    object: String,
    #[serde(default)]
    object_is_entity: bool,
    #[serde(default)]
    object_type: String,
    #[serde(default)]
    statement: String,
    #[serde(default)]
    source_kind: String,
    #[serde(default)]
    confidence: f32,
}

/// Identity used to collapse duplicate memories within and across sessions.
///
/// Entity refs contribute their resolved id, so two surface forms that
/// resolved to the same entity produce the same key; literals are normalized.
fn duplicate_key(memory: &Memory) -> String {
    fn part(r: &EntityRef) -> String {
        match r {
            EntityRef::Entity(id) => format!("e:{id}"),
            EntityRef::Literal(v) => format!("l:{}", v.trim().to_lowercase()),
        }
    }
    format!(
        "{}|{}|{}|{}",
        part(&memory.subject),
        memory.predicate.as_str(),
        part(&memory.object),
        memory.project.as_deref().unwrap_or(""),
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
        // Globals plus this session's project — the same scope the memory
        // will be written into.
        let projects: Vec<String> = transcript.project.clone().into_iter().collect();
        match self.embedder.embed_query(&query).await {
            Ok(vector) => match self
                .memory_repo
                .search_memories_semantic(&vector, None, Some(&projects), PREFETCH_LIMIT)
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

    /// Resolve entities, collapse duplicates, write.
    async fn apply(
        &self,
        transcript: &SessionTranscript,
        raw: RawIngestion,
        prior: &[Memory],
    ) -> Result<IngestionReport, DomainError> {
        let now = unix_now();
        let mut entity_cache: HashMap<String, String> = HashMap::new();
        let mut report = IngestionReport::default();

        // Duplicate index over the prefetched priors, plus everything written
        // in this run so a model that emits the same triple twice writes once.
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
            let predicate = match Predicate::parse(&raw_memory.predicate) {
                Some(p) => p,
                None => {
                    report.predicates_out_of_vocabulary += 1;
                    debug!(
                        "predicate '{}' is outside the vocabulary; recorded as relates_to",
                        raw_memory.predicate.trim()
                    );
                    Predicate::RelatesTo
                }
            };

            let subject = self
                .resolve_ref(
                    &raw_memory.subject,
                    raw_memory.subject_is_entity,
                    &raw_memory.subject_type,
                    now,
                    &mut entity_cache,
                    &mut report,
                )
                .await?;
            let object = self
                .resolve_ref(
                    &raw_memory.object,
                    raw_memory.object_is_entity,
                    &raw_memory.object_type,
                    now,
                    &mut entity_cache,
                    &mut report,
                )
                .await?;

            let memory = Memory {
                id: Uuid::new_v4().to_string(),
                kind: MemoryKind::Fact,
                subject,
                predicate,
                object,
                statement: statement.to_string(),
                project: transcript.project.clone(),
                recorded_at: now,
                source_session_id: Some(transcript.id.clone()),
                source_message_index: None,
                source_kind: SourceKind::parse(&raw_memory.source_kind)
                    .unwrap_or(SourceKind::AssistantInferred),
                confidence: raw_memory.confidence.clamp(0.0, 1.0),
            };

            let key = duplicate_key(&memory);
            if let Some(existing_id) = seen.get(&key).cloned() {
                // Hard delete + insert — newest write wins. No edge, no
                // corroboration bookkeeping.
                self.memory_repo.delete_memory(&existing_id).await?;
                report.memories_deduped += 1;
            }

            let vector = self.embed_opt(&memory.statement).await;
            self.memory_repo
                .append_memory(&memory, vector.as_deref())
                .await?;
            report.memories_written += 1;
            seen.insert(key, memory.id.clone());
        }
        Ok(report)
    }

    /// Resolve a subject/object surface form to an [`EntityRef`].
    ///
    /// Literals pass through. Entity references hit the in-run cache, then an
    /// exact (case-insensitive, role-word-normalized) name-key match, then a
    /// new entity is created. The embedding-similarity and LLM adjudication
    /// tiers the old path used are gone: they were the source of permanent
    /// duplicate anchors, and `entity_name_key` normalization now catches the
    /// variants that matter ("orders-events", "the orders-events service",
    /// "orders events" all share one key).
    async fn resolve_ref(
        &self,
        surface: &str,
        is_entity: bool,
        entity_type: &str,
        now: i64,
        cache: &mut HashMap<String, String>,
        report: &mut IngestionReport,
    ) -> Result<EntityRef, DomainError> {
        let trimmed = surface.trim();
        if !is_entity || trimmed.is_empty() {
            return Ok(EntityRef::Literal(trimmed.to_string()));
        }
        let entity_type = match entity_type.trim() {
            "" => UNKNOWN_ENTITY_TYPE.to_string(),
            t => t.to_lowercase(),
        };
        // Keyed by the same normalization the store uses, so two surface
        // forms one session produces for one thing share a cache slot too.
        let key = format!("{}\u{0}{entity_type}", entity_name_key(trimmed));
        if let Some(id) = cache.get(&key) {
            return Ok(EntityRef::Entity(id.clone()));
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
            return Ok(EntityRef::Entity(existing.id));
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
        Ok(EntityRef::Entity(entity.id))
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
            subject: EntityRef::Entity("entity-1".to_string()),
            predicate: Predicate::Uses,
            object: EntityRef::Literal("svc-a".to_string()),
            statement: "the team uses svc-a".to_string(),
            project: Some("owner/repo".to_string()),
            recorded_at: 100,
            source_session_id: Some("session-1".to_string()),
            source_message_index: None,
            source_kind,
            confidence,
        }
    }

    /// A type conflict outranks any name match. Merging two differently-typed
    /// entities cannot be undone — there is no supersession chain for an
    /// entity — so even a perfect name match must not do it.
    #[test]
    fn a_type_conflict_keeps_entities_apart() {
        assert!(types_conflict("project", "tool"));
        assert!(types_conflict("person", "project"));
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

    /// Predicate spelling can no longer split a duplicate: the vocabulary is
    /// closed, so `Uses`, `utilises` and `depends_on` all *parse* to the same
    /// variant long before the key is built.
    #[test]
    fn predicate_synonyms_resolve_to_one_variant() {
        for spelling in ["uses", "Uses", "  USES ", "utilises", "depends_on", "used"] {
            assert_eq!(
                Predicate::parse(spelling),
                Some(Predicate::Uses),
                "{spelling:?} should fold into `uses`",
            );
        }
        assert_eq!(Predicate::parse("frobnicates"), None);
    }

    #[test]
    fn duplicate_key_ignores_literal_case() {
        let mut a = memory_with("a", SourceKind::UserStated, 0.5);
        let mut b = memory_with("b", SourceKind::Derived, 0.9);
        a.predicate = Predicate::Uses;
        b.predicate = Predicate::Uses;
        a.object = EntityRef::Literal("Svc-A".to_string());
        b.object = EntityRef::Literal("svc-a".to_string());
        assert_eq!(duplicate_key(&a), duplicate_key(&b));
    }

    #[test]
    fn duplicate_key_separates_scopes_and_entity_from_literal() {
        let base = memory_with("a", SourceKind::UserStated, 0.5);
        let mut other_project = base.clone();
        other_project.project = Some("owner/other".to_string());
        assert_ne!(duplicate_key(&base), duplicate_key(&other_project));

        let mut global = base.clone();
        global.project = None;
        assert_ne!(duplicate_key(&base), duplicate_key(&global));

        // A literal "svc-a" and an entity that happens to be named "svc-a"
        // are different memories; collapsing them would merge a resolved
        // reference into an unresolved string.
        let mut as_entity = base.clone();
        as_entity.object = EntityRef::Entity("svc-a".to_string());
        assert_ne!(duplicate_key(&base), duplicate_key(&as_entity));
    }

    #[test]
    fn parses_a_memory_with_entity_type_metadata() {
        let response = r#"{"memories": [
            {"subject": "the user", "subject_is_entity": true, "subject_type": "person",
             "predicate": "prefers", "object": "tabs", "object_is_entity": false,
             "object_type": "unknown", "statement": "the user prefers tabs",
             "source_kind": "user_stated", "confidence": 0.9}
        ]}"#;
        let parsed = parse_ingestion(response).unwrap();
        assert_eq!(parsed.memories.len(), 1);
        let c = &parsed.memories[0];
        assert_eq!(c.subject_type, "person");
        assert!(!c.object_is_entity);
    }

    #[test]
    fn parses_fenced_json() {
        let response = "Sure! Here you go:\n```json\n{\"memories\": [\
            {\"subject\": \"the user\", \"subject_is_entity\": true, \"predicate\": \"prefers\", \
             \"object\": \"tabs\", \"object_is_entity\": false, \
             \"statement\": \"prefers tabs\", \
             \"source_kind\": \"user_stated\", \"confidence\": 0.8}]}\n```\nHope that helps!";
        let parsed = parse_ingestion(response).unwrap();
        assert_eq!(parsed.memories.len(), 1);
    }

    /// Small local models emit markdown escapes that are illegal JSON. The
    /// repair pass is what keeps one stray backslash from costing a whole
    /// session's memory.
    #[test]
    fn repairs_invalid_escapes_before_giving_up() {
        let response = r#"{"memories": [{"subject": "the user", "subject_is_entity": true,
            "predicate": "prefers", "object": "snake\_case", "object_is_entity": false,
            "statement": "the user prefers snake\_case names",
            "source_kind": "user_stated", "confidence": 0.7}]}"#;
        let parsed = parse_ingestion(response).unwrap();
        assert_eq!(parsed.memories.len(), 1);
        assert!(parsed.memories[0].statement.contains("snake"));
    }

    #[test]
    fn unknown_source_kind_falls_back_rather_than_dropping_the_memory() {
        assert_eq!(
            SourceKind::parse("nonsense").unwrap_or(SourceKind::AssistantInferred),
            SourceKind::AssistantInferred,
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
