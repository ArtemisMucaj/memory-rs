//! Persistence port for the append-only memory graph.
//!
//! The store is a log, not a table of current values: an "update" is a *new*
//! memory that closes out an old one through a typed edge, so nothing written by
//! ingestion is ever rewritten in place. That is what makes supersession
//! history, provenance and reversibility answerable — questions a
//! mutate-in-place store destroys the moment it answers them.
//!
//! Two consequences show up directly in these signatures. Retiring a memory is a
//! *status transition* ([`MemoryRepository::set_memory_status`]), never a delete;
//! and the one destructive method
//! ([`MemoryRepository::delete_memories_for_session`]) is scoped to a single
//! session because re-running extraction over an unchanged transcript is a
//! do-over, not a new observation.

use async_trait::async_trait;

use crate::domain::{
    DomainError, EdgeType, Entity, Memory, MemoryEdge, MemoryKind, MemoryStatus, MemoryStoreStats,
};

/// Persistence port for memories, typed edges, and resolved entities.
///
/// Vectors are the embedding of the memory `statement` / entity name; `None`
/// when embeddings are unavailable (the row stays keyword-searchable).
///
/// Scope filters take `projects: Option<&[String]>` rather than a single
/// project because a namespace resolves to *many* projects, and namespace-wide
/// recall is a shipped feature. `None` means every scope; `Some(&[])` means
/// globals only.
#[async_trait]
pub trait MemoryRepository: Send + Sync {
    // ── Memories (append-only log) ─────────────────────────────────────────

    /// Append a memory to the log, with its optional statement embedding.
    ///
    /// The memory id is expected to be unique; appending a memory whose id
    /// already exists replaces that row (idempotent re-append), but callers on
    /// the ingestion path always mint a fresh id — an "update" is a new memory.
    async fn append_memory(
        &self,
        memory: &Memory,
        vector: Option<&[f32]>,
    ) -> Result<(), DomainError>;

    /// Fetch a memory by id.
    async fn find_memory(&self, id: &str) -> Result<Option<Memory>, DomainError>;

    /// Fetch many memories by id in one query.
    ///
    /// Recall walks enrichment edges and then needs the memories those edges
    /// point at; resolving them one call at a time is an N+1 against a
    /// connection shared with nodes, sessions and digests. Unknown ids are
    /// skipped rather than erroring, and the result order is unspecified.
    async fn find_memories(&self, ids: &[String]) -> Result<Vec<Memory>, DomainError>;

    /// List memories, newest first, optionally restricted by `kind`, lifecycle
    /// `status`, and scope (`projects` plus globals).
    async fn list_memories(
        &self,
        kind: Option<MemoryKind>,
        status: Option<MemoryStatus>,
        projects: Option<&[String]>,
    ) -> Result<Vec<Memory>, DomainError>;

    /// Transition a memory's lifecycle status, optionally closing its validity
    /// window (`valid_to`). Returns whether the memory existed. This is the
    /// non-destructive way an append-only store retires a memory (e.g. flipping
    /// it to `superseded` when a newer memory supersedes it).
    ///
    /// `valid_to: None` leaves any existing `valid_to` untouched — retracting a
    /// memory must not silently rewrite when it stopped being true. Use
    /// [`Self::reopen_memory`] to actually clear it.
    async fn set_memory_status(
        &self,
        id: &str,
        status: MemoryStatus,
        valid_to: Option<i64>,
    ) -> Result<bool, DomainError>;

    /// Overwrite a memory's confidence, returning whether the memory existed.
    ///
    /// Not an append-only violation, for the same reason
    /// [`Self::set_memory_status`] is not: confidence is lifecycle metadata
    /// *about* an assertion, not the assertion itself. The memory's statement,
    /// its embedding and its recall behaviour are untouched.
    ///
    /// It exists as its own method because the alternative — reading the memory,
    /// editing the field and re-appending it — routes through
    /// [`Self::append_memory`], which replaces the row *and its vector*. A
    /// caller that re-appended without re-embedding would silently drop the
    /// memory out of semantic recall as a side effect of nudging a number.
    async fn set_memory_confidence(&self, id: &str, confidence: f32) -> Result<bool, DomainError>;

    /// Return a memory to `active` *and* clear its `valid_to`.
    ///
    /// The consolidation pass can decide that a supersession was wrong — that
    /// the older memory is in fact still true. Restoring it needs to clear the
    /// validity window, which [`Self::set_memory_status`] deliberately will not
    /// do: a reopened memory carrying a stale `valid_to` would assert that it
    /// stopped being true at a moment it did not.
    async fn reopen_memory(&self, id: &str) -> Result<bool, DomainError>;

    /// Hard-delete every memory whose provenance is `session_id`, along with its
    /// vector and any edges touching it. The single sanctioned destructive
    /// operation, used only by a forced re-import: re-running extraction over an
    /// unchanged transcript is a do-over, not a new observation, so the prior
    /// run's memories are wiped rather than tombstoned. Returns the number of
    /// memories removed.
    async fn delete_memories_for_session(&self, session_id: &str) -> Result<usize, DomainError>;

    /// How many memories currently carry `session_id` as their provenance. Used
    /// as the ingestion idempotence marker: a non-forced re-ingest of a session
    /// that already produced memories is skipped.
    async fn count_memories_for_session(&self, session_id: &str) -> Result<usize, DomainError>;

    // ── Typed edges ──────────────────────────────────────────────────────

    /// Insert (or replace) a typed edge between two memories, keyed by
    /// `(from, to, type)`.
    async fn add_edge(&self, edge: &MemoryEdge) -> Result<(), DomainError>;

    /// Edges originating at `memory_id`.
    async fn edges_from(&self, memory_id: &str) -> Result<Vec<MemoryEdge>, DomainError>;

    /// Edges pointing at `memory_id`.
    async fn edges_to(&self, memory_id: &str) -> Result<Vec<MemoryEdge>, DomainError>;

    /// Every edge touching any of `ids`, in either direction, in one query.
    ///
    /// Recall now returns each hit's provenance — what it superseded, what
    /// corroborates it, what it still contradicts — which means edges are
    /// needed for a whole result set at once. Doing that with `edges_from` +
    /// `edges_to` per hit is two round-trips per result on a connection shared
    /// with nodes, sessions and digests. Duplicate ids are tolerated; the
    /// result order is unspecified.
    async fn edges_for(&self, ids: &[String]) -> Result<Vec<MemoryEdge>, DomainError>;

    /// Every edge of one type, or every edge when `edge_type` is `None`.
    ///
    /// This is how the conflict queue is derived rather than stored: the
    /// unresolved conflicts are the `contradicts` edges whose two endpoints are
    /// both still active. A memory therefore cannot get *stuck* in a conflicted
    /// state — resolving one simply means one endpoint stops being active.
    async fn list_edges(&self, edge_type: Option<EdgeType>)
        -> Result<Vec<MemoryEdge>, DomainError>;

    // ── Entities ─────────────────────────────────────────────────────────

    /// Insert or replace an entity (and its aliases), keyed by id, with its
    /// optional name embedding.
    async fn upsert_entity(
        &self,
        entity: &Entity,
        vector: Option<&[f32]>,
    ) -> Result<(), DomainError>;

    /// Fetch an entity by id.
    async fn find_entity(&self, id: &str) -> Result<Option<Entity>, DomainError>;

    /// Resolve an entity by an exact (case-insensitive) match on its canonical
    /// name or any alias. The cheap first leg of entity resolution.
    async fn find_entity_by_alias(&self, alias: &str) -> Result<Option<Entity>, DomainError>;

    /// List all entities, newest first.
    async fn list_entities(&self) -> Result<Vec<Entity>, DomainError>;

    /// Cosine-similarity search over entity name embeddings — the fuzzy second
    /// leg of entity resolution. Returns `(entity, score)` best first.
    async fn search_entities_semantic(
        &self,
        vector: &[f32],
        limit: usize,
    ) -> Result<Vec<(Entity, f32)>, DomainError>;

    /// Move every memory reference from entity `from` to entity `to`, returning
    /// how many references moved (subject and object both count).
    ///
    /// This is how consolidation merges two entities that ingestion created
    /// separately. It is *not* a breach of the append-only rule: a memory's
    /// assertion — its statement, and therefore its embedding and its recall
    /// behaviour — is untouched, and only an internal foreign key is repaired.
    async fn repoint_entity(&self, from: &str, to: &str) -> Result<usize, DomainError>;

    /// Delete an entity along with its aliases and vector. Only safe once
    /// [`Self::repoint_entity`] has moved its memories elsewhere.
    async fn delete_entity(&self, id: &str) -> Result<bool, DomainError>;

    // ── Memory retrieval (entry-point finder) ─────────────────────────────

    /// Cosine-similarity search over `active` memory statement embeddings,
    /// filtered by `kind` and scope. Returns `(memory, score)` best first, score
    /// in `[0, 1]`.
    async fn search_memories_semantic(
        &self,
        vector: &[f32],
        kind: Option<MemoryKind>,
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<(Memory, f32)>, DomainError>;

    /// Case-insensitive keyword search over `active` memory statements, filtered
    /// by `kind` and scope. Returns `(memory, score)` best first.
    async fn search_memories_keyword(
        &self,
        query: &str,
        kind: Option<MemoryKind>,
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<(Memory, f32)>, DomainError>;

    /// Stored embedding for every memory that has one, as `(memory_id, vector)`.
    /// Memories without a vector are omitted. Used by consolidation to cluster
    /// near-duplicate memories.
    async fn list_memory_embeddings(&self) -> Result<Vec<(String, Vec<f32>)>, DomainError>;

    /// Aggregate store statistics.
    async fn memory_stats(&self) -> Result<MemoryStoreStats, DomainError>;
}
