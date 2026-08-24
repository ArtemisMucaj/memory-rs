//! Persistence port for facts (memories), entities, and resources.
//!
//! The store is a table of current values, not a log: updates are hard
//! delete + insert. There is no lifecycle status, no validity window, and no
//! typed edges between memories. Newest write wins.

use async_trait::async_trait;

use crate::domain::{DomainError, Entity, Memory, MemoryResource};

/// Persistence port for memories, entities, and resources.
///
/// Vectors are the embedding of the memory `statement` (or resource
/// `abstract + overview`); `None` when embeddings are unavailable (the row
/// stays keyword-searchable).
///
/// Scope filters take `projects: Option<&[String]>` rather than a single
/// project because a namespace resolves to *many* projects. `None` means
/// every scope; `Some(&[])` means globals only.
#[async_trait]
pub trait MemoryRepository: Send + Sync {
    // ── Memories ─────────────────────────────────────────────────────────

    /// Insert (or replace by id) a memory with its optional statement embedding.
    async fn append_memory(
        &self,
        memory: &Memory,
        vector: Option<&[f32]>,
    ) -> Result<(), DomainError>;

    /// Fetch a memory by id.
    async fn find_memory(&self, id: &str) -> Result<Option<Memory>, DomainError>;

    /// Fetch many memories by id in one query. Unknown ids are skipped.
    async fn find_memories(&self, ids: &[String]) -> Result<Vec<Memory>, DomainError>;

    /// Newest first, optionally filtered by project scope. The `kind`
    /// parameter is gone: there is only one kind.
    async fn list_memories(&self, projects: Option<&[String]>) -> Result<Vec<Memory>, DomainError>;

    /// Hard-delete one memory and its embedding. Returns whether it existed.
    async fn delete_memory(&self, id: &str) -> Result<bool, DomainError>;

    /// Hard-delete every memory extracted from `session_id`, along with its
    /// embedding. Used by forced re-import: re-running extraction over an
    /// unchanged transcript is a do-over, not a new observation. Returns the
    /// count removed.
    async fn delete_memories_for_session(&self, session_id: &str) -> Result<usize, DomainError>;

    /// How many memories currently carry `session_id` as their provenance.
    /// Used as the ingestion idempotence marker: a non-forced re-ingest of a
    /// session that already produced memories is skipped.
    async fn count_memories_for_session(&self, session_id: &str) -> Result<usize, DomainError>;

    // ── Entities ─────────────────────────────────────────────────────────

    /// Insert or replace an entity (and every name it goes by), keyed by id.
    async fn upsert_entity(&self, entity: &Entity) -> Result<(), DomainError>;

    /// Fetch an entity by id.
    async fn find_entity(&self, id: &str) -> Result<Option<Entity>, DomainError>;

    /// Fetch many entities by id in one query. Unknown ids are skipped.
    async fn find_entities(&self, ids: &[String]) -> Result<Vec<Entity>, DomainError>;

    /// Every entity whose `name_key` matches `entity_name_key(name)` — the
    /// only resolution tier. No embeddings, no LLM adjudication.
    async fn find_entities_by_name(&self, name: &str) -> Result<Vec<Entity>, DomainError>;

    /// List all entities, newest first.
    async fn list_entities(&self) -> Result<Vec<Entity>, DomainError>;

    /// Memories referencing `entity_id` as subject or object, newest first.
    async fn memories_for_entity(&self, entity_id: &str) -> Result<Vec<Memory>, DomainError>;

    // ── Retrieval ────────────────────────────────────────────────────────

    /// Cosine-similarity search over memory statement embeddings. Returns
    /// `(memory, score)` best first.
    async fn search_memories_semantic(
        &self,
        vector: &[f32],
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<(Memory, f32)>, DomainError>;

    /// Case-insensitive keyword search over memory statements. Returns
    /// `(memory, score)` best first.
    async fn search_memories_keyword(
        &self,
        query: &str,
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<(Memory, f32)>, DomainError>;

    /// All memories matching the filters, newest first. Used as the recency
    /// leg of recall's RRF fusion.
    async fn list_memories_by_recency(
        &self,
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<Memory>, DomainError>;

    // ── Resources ────────────────────────────────────────────────────────

    /// Insert or replace a resource and its embedding.
    async fn upsert_resource(
        &self,
        resource: &MemoryResource,
        vector: Option<&[f32]>,
    ) -> Result<(), DomainError>;

    async fn find_resource(&self, uri: &str) -> Result<Option<MemoryResource>, DomainError>;

    async fn list_resources(&self) -> Result<Vec<MemoryResource>, DomainError>;

    /// Cosine similarity over resource embeddings. Returns `(resource, score)`
    /// best first.
    async fn search_resources_semantic(
        &self,
        vector: &[f32],
        limit: usize,
    ) -> Result<Vec<(MemoryResource, f32)>, DomainError>;

    // ── Namespaces ───────────────────────────────────────────────────────

    async fn create_namespace(&self, name: &str) -> Result<bool, DomainError>;
    async fn delete_namespace(&self, name: &str) -> Result<bool, DomainError>;
    async fn list_namespaces(&self) -> Result<Vec<(String, u64)>, DomainError>;
    async fn namespace_projects(&self, name: &str) -> Result<Vec<String>, DomainError>;
    async fn namespace_created_at(&self, name: &str) -> Result<Option<i64>, DomainError>;
    async fn assign_project(&self, namespace: &str, project: &str) -> Result<bool, DomainError>;
    async fn unassign_project(&self, namespace: &str, project: &str) -> Result<bool, DomainError>;

    // ── Sessions ─────────────────────────────────────────────────────────

    /// Record an imported session, replacing any prior row with the same
    /// `(source, id)` identity.
    async fn record_session(
        &self,
        session: &crate::domain::ImportedSession,
    ) -> Result<(), DomainError>;

    /// Whether a session has already been imported (or attempted), keyed by
    /// its `(source, id)` pair. Used by the dream harvest to skip re-work.
    /// The composite key matters because Claude, OpenCode and Zed mint ids
    /// from independent namespaces — `id` alone is not unique.
    async fn find_session(
        &self,
        source: &str,
        id: &str,
    ) -> Result<Option<crate::domain::ImportedSession>, DomainError>;

    /// List sessions, newest first.
    async fn list_sessions(
        &self,
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<crate::domain::ImportedSession>, DomainError>;

    /// List sessions with a `status` filter applied in SQL (before `LIMIT`
    /// cuts the window). The dream harvest uses this to skip failed-import
    /// markers without letting them crowd out real sessions.
    async fn list_sessions_by_status(
        &self,
        status: crate::domain::SessionStatus,
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<crate::domain::ImportedSession>, DomainError>;
}
