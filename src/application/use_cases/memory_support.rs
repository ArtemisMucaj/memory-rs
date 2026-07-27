//! Helpers shared by the memory write paths (per-session extraction and the
//! dream cycle), so the embedding recipe and the identity-preserving update
//! semantics are defined exactly once.

use tracing::warn;

use crate::application::interfaces::{Embedder, MemoryRepository};
use crate::domain::{DomainError, MemoryItem, MemoryKind};

/// Outcome of embedding a memory item, distinguishing an intentional no-vector
/// (embeddings switched off) from a transient failure. The two must be handled
/// differently on update: `Disabled` means "no vector by design", while
/// `Failed` must not silently drop an item's existing vector from recall.
pub(crate) enum ItemEmbedding {
    /// A fresh embedding to store.
    Ready(Vec<f32>),
    /// Embeddings are turned off — write no vector.
    Disabled,
    /// Embedding was attempted and failed — keep any existing vector.
    Failed,
}

/// Embed `name + content` for semantic recall, distinguishing "disabled" from
/// "failed" so callers can preserve an existing vector on a transient failure.
pub(crate) async fn embed_memory_item(embedder: &Embedder, item: &MemoryItem) -> ItemEmbedding {
    if !embedder.embeddings_enabled() {
        return ItemEmbedding::Disabled;
    }
    let text = format!("{}\n\n{}", item.name().replace('_', " "), item.content());
    match embedder.embed_query(&text).await {
        Ok(vector) => ItemEmbedding::Ready(vector),
        Err(e) => {
            warn!("failed to embed memory item '{}': {e}", item.name());
            ItemEmbedding::Failed
        }
    }
}

/// Write one upsert, preserving the target's identity and history when it
/// already exists (same id, original `created_at`, bumped `update_count`).
///
/// `source_override` stamps the written item's source session; `None` keeps
/// the existing item's source (or leaves a new item unsourced).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn upsert_preserving_identity(
    memory_repo: &dyn MemoryRepository,
    embedder: &Embedder,
    kind: MemoryKind,
    name: &str,
    content: &str,
    project: Option<String>,
    source_override: Option<&str>,
    now: i64,
) -> Result<(), DomainError> {
    // Scoped to `project`: a same-named memory belonging to another project is a
    // different item and must not be found (let alone rewritten) here.
    let existing = memory_repo
        .find_item(kind, name, project.as_deref())
        .await?;
    let item = match existing {
        Some(prev) => MemoryItem::new(
            prev.id().to_string(),
            kind,
            name.to_string(),
            content.to_string(),
            source_override
                .or(prev.source_session_id())
                .map(str::to_string),
            project,
            prev.created_at(),
            now,
            prev.update_count() + 1,
        ),
        None => MemoryItem::new(
            uuid::Uuid::new_v4().to_string(),
            kind,
            name.to_string(),
            content.to_string(),
            source_override.map(str::to_string),
            project,
            now,
            now,
            0,
        ),
    };
    // `upsert_item` clears any prior vector and only re-inserts the one passed
    // in, so a transient embedding failure must not fall through as `None` —
    // that would permanently drop an updated item from semantic recall. On
    // failure, carry the item's existing stored vector forward instead (`item`
    // reuses the previous id when updating; a brand-new item simply has none).
    let vector = match embed_memory_item(embedder, &item).await {
        ItemEmbedding::Ready(vector) => Some(vector),
        ItemEmbedding::Disabled => None,
        ItemEmbedding::Failed => memory_repo.find_item_vector(item.id()).await?,
    };
    memory_repo.upsert_item(&item, vector.as_deref()).await
}

/// Current Unix time in seconds.
pub(crate) fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
