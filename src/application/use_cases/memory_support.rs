//! Helpers shared by the memory write paths (per-session extraction and the
//! dream cycle), so the embedding recipe and the identity-preserving update
//! semantics are defined exactly once.

use tracing::warn;

use crate::application::interfaces::{Embedder, NodeRepository};
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
///
/// `writer` says how much authority the caller has to change an existing
/// memory's scope — see [`WriteScope`] and [`resolve_scope`].
#[allow(clippy::too_many_arguments)]
pub(crate) async fn upsert_preserving_identity(
    node_repo: &dyn NodeRepository,
    embedder: &Embedder,
    kind: MemoryKind,
    name: &str,
    content: &str,
    project: Option<String>,
    writer: WriteScope<'_>,
    source_override: Option<&str>,
    now: i64,
) -> Result<(), DomainError> {
    // Resolved across *every* scope, not just the requested one: a `(kind,
    // name)` may not exist both globally and under a project, so the write has
    // to see a same-named item at the other scope in order to reuse it rather
    // than add a sibling.
    let candidates = node_repo.find_items_named(kind, name).await?;
    let resolved = resolve_scope(&candidates, project, writer);
    let project = resolved.project;
    let superseded = resolved.superseded;
    let existing = resolved.target.cloned();
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
        ItemEmbedding::Failed => node_repo.find_item_vector(item.id()).await?,
    };
    node_repo.upsert_item(&item, vector.as_deref()).await?;

    // Folding project rows into a global one is only half the job: the rows
    // that were absorbed have to go, or the store ends up holding exactly the
    // global/project pair this whole path exists to prevent. Done after the
    // write so a failure mid-way leaves duplicates (recoverable) rather than a
    // hole where the memory used to be.
    for id in superseded {
        node_repo.delete_item_by_id(&id).await?;
    }
    Ok(())
}

/// How much authority a writer has to change an existing memory's scope.
///
/// Both variants write through the same path; they differ only in what they are
/// allowed to do when a same-named memory already exists under a project the
/// writer did not observe.
#[derive(Clone, Copy)]
pub(crate) enum WriteScope<'a> {
    /// A per-session extraction. Carries the session's project, which is *not*
    /// the scope the model asked for — a session in project X routinely emits
    /// global items. Evidence is limited to that one session, so this writer
    /// never generalises over projects it never saw.
    Session(Option<&'a str>),
    /// The dream cycle. It reads every project at once and exists to merge
    /// across them, so it may fold project-scoped rows into a global one.
    Consolidation,
}

/// Where an upsert should land: which row it rewrites, at what scope, and which
/// same-named rows it replaces.
struct Resolution<'a> {
    /// The existing row to rewrite (keeping its id and history), if any.
    target: Option<&'a MemoryItem>,
    /// The scope to write at — `None` is global.
    project: Option<String>,
    /// Ids of rows this write absorbs; deleted once it lands.
    superseded: Vec<String>,
}

impl<'a> Resolution<'a> {
    /// Rewrite `target` at `project`, absorbing nothing.
    fn at(target: Option<&'a MemoryItem>, project: Option<String>) -> Self {
        Self {
            target,
            project,
            superseded: Vec::new(),
        }
    }
}

/// Pick which existing item an upsert rewrites, and the scope it lands at.
///
/// **The invariant:** a given `(kind, name)` is *either* global *or* owned by
/// one or more projects — never both. Two projects holding the same name stay
/// legal (widening the key to include `project` was itself a fix: a memory
/// extracted in project B used to overwrite A's and relabel it as B's). What is
/// not legal is a global row and a project row for the same name, because
/// recall scoped to that project returns both — two entries, same name,
/// potentially contradicting each other, with nothing to say which wins.
///
/// SQL cannot express "either global or project-scoped" as a `UNIQUE`, so this
/// is the enforcement point: every memory write funnels through here.
///
/// | requested | existing        | outcome                                  |
/// |-----------|-----------------|------------------------------------------|
/// | project p | row for p       | rewrite it (the ordinary update)          |
/// | project p | global row      | rewrite the **global** row, stay global   |
/// | global    | row for session | **promote** that row to global (a move)   |
/// | global    | other projects' | stay scoped to the session's project      |
fn resolve_scope<'a>(
    candidates: &'a [MemoryItem],
    requested: Option<String>,
    writer: WriteScope<'_>,
) -> Resolution<'a> {
    let global = candidates.iter().find(|item| item.project().is_none());
    let find = |project: &str| {
        candidates
            .iter()
            .find(|item| item.project() == Some(project))
    };

    match requested {
        // Asked for a project scope. An existing row for that same project is
        // the obvious target; otherwise a global row wins and *keeps* its
        // reach. A session re-stating a general memory is not evidence that the
        // memory is specific to that session's project, so narrowing it here
        // would strip it from every other project on the strength of one
        // observation. If it really is project-specific the content will say
        // so, and the dream cycle — which sees every project at once — can
        // split it with far better evidence than this one write has.
        Some(project) => match find(&project) {
            Some(item) => Resolution::at(Some(item), Some(project)),
            None => match global {
                Some(item) => Resolution::at(Some(item), None),
                None => Resolution::at(None, Some(project)),
            },
        },
        // Asked for global, and a global row already exists: the ordinary
        // update, and by the invariant there is nothing else to absorb.
        None if global.is_some() => Resolution::at(global, None),
        // Asked for global with nothing in the way: an ordinary new item.
        None if candidates.is_empty() => Resolution::at(None, None),
        // Asked for global, but some project already owns this name. Only the
        // dream cycle may resolve that by generalising; a single session has
        // seen one project and cannot speak for the others.
        None => {
            if let WriteScope::Session(session_project) = writer {
                return match session_project {
                    // The session owns a row here. If every row in the way is
                    // its own, it has the standing to generalise and the row is
                    // promoted; if another project also holds the name, the
                    // write stays in the session's own lane.
                    Some(project) => {
                        let foreign = candidates
                            .iter()
                            .any(|item| item.project() != Some(project));
                        if foreign {
                            Resolution::at(find(project), Some(project.to_string()))
                        } else {
                            Resolution::at(find(project), None)
                        }
                    }
                    // A session with no project of its own, and the name
                    // belongs to projects it never saw. It has no lane to stay
                    // in and no standing to speak for them, so it refreshes the
                    // strongest existing row in place rather than adding a
                    // global twin beside it.
                    None => {
                        let target = candidates.iter().max_by_key(|item| item.update_count());
                        let project = target.and_then(MemoryItem::project).map(str::to_string);
                        Resolution::at(target, project)
                    }
                };
            }
            // The dream cycle: it reads every project at once, and folding them
            // into one general item is the whole point of the pass.
            //
            // Reuse the most-updated row as the target so the surviving memory
            // keeps the longest history rather than restarting at zero, and
            // mark the rest superseded — they are what this item now says.
            let target = candidates.iter().max_by_key(|item| item.update_count());
            let superseded = candidates
                .iter()
                .filter(|item| Some(item.id()) != target.map(MemoryItem::id))
                .map(|item| item.id().to_string())
                .collect();
            Resolution {
                target,
                project: None,
                superseded,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stored item under `project` (`None` = global) with `updates` history.
    fn item(id: &str, project: Option<&str>, updates: u32) -> MemoryItem {
        MemoryItem::new(
            id.to_string(),
            MemoryKind::Fact,
            "build_command".to_string(),
            "content".to_string(),
            None,
            project.map(str::to_string),
            0,
            0,
            updates,
        )
    }

    fn resolve<'a>(
        candidates: &'a [MemoryItem],
        requested: Option<&str>,
        writer: WriteScope<'_>,
    ) -> Resolution<'a> {
        resolve_scope(candidates, requested.map(str::to_string), writer)
    }

    #[test]
    fn project_write_rewrites_the_global_item_and_keeps_it_global() {
        let candidates = [item("g", None, 7)];
        let resolved = resolve(
            &candidates,
            Some("svc-a"),
            WriteScope::Session(Some("svc-a")),
        );
        // The regression this whole path exists for: this used to miss the
        // global row and insert a project sibling next to it.
        assert_eq!(resolved.target.map(MemoryItem::id), Some("g"));
        assert_eq!(resolved.project, None);
        assert!(resolved.superseded.is_empty());
    }

    #[test]
    fn global_write_promotes_the_sessions_own_project_item() {
        let candidates = [item("a", Some("svc-a"), 3)];
        let resolved = resolve(&candidates, None, WriteScope::Session(Some("svc-a")));
        // A move, not an insert: same row, history intact, now global.
        assert_eq!(resolved.target.map(MemoryItem::id), Some("a"));
        assert_eq!(resolved.project, None);
        assert!(resolved.superseded.is_empty());
    }

    #[test]
    fn global_write_stays_scoped_when_another_project_owns_the_name() {
        let candidates = [item("b", Some("svc-b"), 3)];
        let resolved = resolve(&candidates, None, WriteScope::Session(Some("svc-a")));
        // svc-a never observed svc-b, so it may not speak for it.
        assert_eq!(resolved.target.map(MemoryItem::id), None);
        assert_eq!(resolved.project.as_deref(), Some("svc-a"));
        assert!(resolved.superseded.is_empty());
    }

    #[test]
    fn consolidation_folds_project_items_into_one_global_item() {
        let candidates = [item("a", Some("svc-a"), 1), item("b", Some("svc-b"), 5)];
        let resolved = resolve(&candidates, None, WriteScope::Consolidation);
        // Dream sees every project, so it may generalise. It keeps the longest
        // history and absorbs the rest rather than leaving them beside it.
        assert_eq!(resolved.target.map(MemoryItem::id), Some("b"));
        assert_eq!(resolved.project, None);
        assert_eq!(resolved.superseded, vec!["a".to_string()]);
    }

    #[test]
    fn projectless_session_never_absorbs_another_projects_item() {
        let candidates = [item("a", Some("svc-a"), 1), item("b", Some("svc-b"), 5)];
        let resolved = resolve(&candidates, None, WriteScope::Session(None));
        // Only the dream cycle may generalise across projects. A session with
        // no project has no lane to stay in, but that is not licence to delete
        // memories belonging to projects it never saw.
        assert_eq!(resolved.target.map(MemoryItem::id), Some("b"));
        assert_eq!(resolved.project.as_deref(), Some("svc-b"));
        assert!(resolved.superseded.is_empty());
    }

    #[test]
    fn distinct_projects_keep_their_own_items() {
        let candidates = [item("a", Some("svc-a"), 1), item("b", Some("svc-b"), 1)];
        let resolved = resolve(
            &candidates,
            Some("svc-b"),
            WriteScope::Session(Some("svc-b")),
        );
        // Widening the key to include `project` was itself a fix; a write for
        // svc-b must never land on svc-a's row.
        assert_eq!(resolved.target.map(MemoryItem::id), Some("b"));
        assert_eq!(resolved.project.as_deref(), Some("svc-b"));
        assert!(resolved.superseded.is_empty());
    }
}
