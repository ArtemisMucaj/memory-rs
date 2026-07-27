use async_trait::async_trait;

use crate::domain::{
    DomainError, DreamRun, ImportedSession, MemoryItem, MemoryKind, MemoryNode, NodeKind,
};

/// Persistence port for long-term memory items and imported-session records.
///
/// Memory lives in its own store (a dedicated DuckDB file, separate from the
/// code index) so that importing sessions never contends with indexing and
/// the memory database can be inspected, backed up, or wiped independently.
#[async_trait]
pub trait MemoryRepository: Send + Sync {
    /// Insert or replace a memory item, keyed by `(kind, name, project)`.
    ///
    /// `vector` is the embedding of the item content; `None` when embeddings
    /// are unavailable (the item remains keyword-searchable).
    async fn upsert_item(
        &self,
        item: &MemoryItem,
        vector: Option<&[f32]>,
    ) -> Result<(), DomainError>;

    /// Find the item with exactly this identity. `project: None` addresses the
    /// global item, which is a *different* item from a same-named one scoped to
    /// a project — two projects that independently learn "flaky_test_fix" keep
    /// separate memories rather than overwriting each other.
    async fn find_item(
        &self,
        kind: MemoryKind,
        name: &str,
        project: Option<&str>,
    ) -> Result<Option<MemoryItem>, DomainError>;

    /// Every item sharing `(kind, name)` across all project scopes, newest
    /// first. For callers holding a `kind/name` reference with no project to
    /// disambiguate it (CLI `show`/`delete`, model-issued deletes).
    async fn find_items_named(
        &self,
        kind: MemoryKind,
        name: &str,
    ) -> Result<Vec<MemoryItem>, DomainError>;

    /// Find an item by its ID.
    async fn find_item_by_id(&self, id: &str) -> Result<Option<MemoryItem>, DomainError>;

    /// Delete by the full `(kind, name, project)` identity. Returns `true` when
    /// an item was removed.
    async fn delete_item(
        &self,
        kind: MemoryKind,
        name: &str,
        project: Option<&str>,
    ) -> Result<bool, DomainError>;

    /// Delete by item ID. Returns `true` when an item was removed.
    async fn delete_item_by_id(&self, id: &str) -> Result<bool, DomainError>;

    /// List items, optionally restricted to one kind, newest first.
    async fn list_items(&self, kind: Option<MemoryKind>) -> Result<Vec<MemoryItem>, DomainError>;

    /// Cosine-similarity search over item embeddings.
    /// Returns `(item, score)` pairs, best first, score in `[0, 1]`.
    ///
    /// `project` filters to items relevant in that project/namespace —
    /// global items plus items belonging to exactly that project. `None`
    /// searches everything.
    async fn search_semantic(
        &self,
        vector: &[f32],
        kind: Option<MemoryKind>,
        project: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(MemoryItem, f32)>, DomainError>;

    /// Case-insensitive keyword search over item names and content.
    /// Returns `(item, score)` pairs, best first. `project` filters as in
    /// [`Self::search_semantic`].
    async fn search_keyword(
        &self,
        query: &str,
        kind: Option<MemoryKind>,
        project: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(MemoryItem, f32)>, DomainError>;

    /// Stored embedding for every item that has one, as `(item_id, vector)`.
    /// Items without a vector (embeddings disabled at write time) are omitted.
    /// Used by dream consolidation to cluster near-duplicate memories.
    async fn list_item_vectors(&self) -> Result<Vec<(String, Vec<f32>)>, DomainError>;

    /// Stored embedding for a single item by ID, or `None` if it has none.
    /// Used to preserve an item's existing vector across an update whose
    /// re-embedding transiently failed, so it is not dropped from recall.
    async fn find_item_vector(&self, id: &str) -> Result<Option<Vec<f32>>, DomainError>;

    /// Record that a session has been imported (idempotence marker).
    async fn record_session(&self, session: &ImportedSession) -> Result<(), DomainError>;

    async fn find_session(&self, id: &str) -> Result<Option<ImportedSession>, DomainError>;

    /// List imported sessions, newest first.
    async fn list_sessions(&self) -> Result<Vec<ImportedSession>, DomainError>;

    // ── Virtual filesystem nodes (L0/L1/L2) ──────────────────────────────

    /// Insert or replace a node, keyed by its `uri`.
    ///
    /// `vector` is the embedding of the node's L0/L1 summary; `None` when
    /// embeddings are unavailable (the node remains keyword-searchable and
    /// browsable by URI).
    async fn upsert_node(
        &self,
        node: &MemoryNode,
        vector: Option<&[f32]>,
    ) -> Result<(), DomainError>;

    /// Fetch a single node by its `memory://` URI.
    async fn find_node(&self, uri: &str) -> Result<Option<MemoryNode>, DomainError>;

    /// Delete a node (and its embedding) by URI. Returns whether it existed.
    async fn delete_node(&self, uri: &str) -> Result<bool, DomainError>;

    /// List the direct children of a directory URI (its immediate members in
    /// the virtual filesystem), newest first.
    async fn list_child_nodes(&self, parent_uri: &str) -> Result<Vec<MemoryNode>, DomainError>;

    /// List nodes, optionally restricted to one kind, newest first.
    async fn list_nodes(&self, kind: Option<NodeKind>) -> Result<Vec<MemoryNode>, DomainError>;

    /// Cosine-similarity search over node L0/L1 embeddings.
    /// Returns `(node, score)` pairs, best first, score in `[0, 1]`.
    async fn search_nodes_semantic(
        &self,
        vector: &[f32],
        kind: Option<NodeKind>,
        limit: usize,
    ) -> Result<Vec<(MemoryNode, f32)>, DomainError>;

    /// Case-insensitive keyword search over node abstracts and overviews.
    /// Returns `(node, score)` pairs, best first.
    async fn search_nodes_keyword(
        &self,
        query: &str,
        kind: Option<NodeKind>,
        limit: usize,
    ) -> Result<Vec<(MemoryNode, f32)>, DomainError>;

    // ── Dream runs ───────────────────────────────────────────────────────

    /// Record a completed dream cycle.
    async fn record_dream_run(&self, run: &DreamRun) -> Result<(), DomainError>;

    /// The most recently finished dream run, if any.
    async fn last_dream_run(&self) -> Result<Option<DreamRun>, DomainError>;

    /// Aggregate memory-store statistics: item counts by kind, session count,
    /// and node counts by kind.
    async fn stats(&self) -> Result<MemoryStats, DomainError>;

    /// The embedding model that wrote the stored vectors, persisted on first
    /// open. This is authoritative for **retrieval**: queries must be embedded
    /// with the same model the vectors were, or the cosine comparison is
    /// meaningless. Empty only for a store that has never recorded one.
    async fn embedding_model(&self) -> Result<String, DomainError>;
}

/// Statistics about the memory store.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total memory items across all kinds.
    pub total_items: u64,
    /// Breakdown of memory items by kind.
    pub items_by_kind: Vec<(String, u64)>,
    /// Total imported sessions.
    pub total_sessions: u64,
    /// Total nodes across all kinds.
    pub total_nodes: u64,
    /// Breakdown of nodes by kind.
    pub nodes_by_kind: Vec<(String, u64)>,
}
