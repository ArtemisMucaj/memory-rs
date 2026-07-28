//! Shared controllers — the single source of truth for every memory operation.
//!
//! Each function takes the [`Container`] plus typed parameters and returns
//! **domain data** (items, nodes, stats, outcome enums), performing the logic
//! common to all surfaces — scope resolution for a namespace, the
//! `kind/name`-or-id reference resolution for show/delete, and so on — but *no*
//! presentation. The CLI router, the HTTP management API, and the MCP server
//! all call these, then render the result in their own format. This keeps the
//! three surfaces in lockstep with zero duplicated logic.

use crate::application::{
    resource_slug, DreamReport, ImportOutcome, IngestionOutcome, MemorySearchUseCase, NodeStats,
    Recalled, MEMORY_ROOT_URI, RESOURCES_ROOT_URI, SESSIONS_ROOT_URI,
};
use crate::connector::adapter::{fetch_resource, parse_transcript_file};
use crate::connector::api::Container;
use crate::domain::{
    DomainError, EdgeType, ImportedSession, Memory, MemoryEdge, MemoryItem, MemoryKind, MemoryNode,
    MemoryStatus, MemoryStoreStats, NodeKind,
};

/// How a memory should be scoped for a search.
#[derive(Debug, Clone, Default)]
pub enum SearchScope {
    /// Everything (globals + all projects).
    #[default]
    All,
    /// Globals plus one project.
    Project(String),
    /// Globals plus a namespace's member projects.
    Namespace(String),
}

/// Resolve a [`SearchScope`] into the concrete project list the repository
/// expects: `None` = all; `Some(list)` = globals + `list`. A namespace with no
/// members resolves to [`ScopeResolution::EmptyNamespace`] so the caller can
/// tell the user rather than silently searching only globals.
pub enum ScopeResolution {
    All,
    Projects(Vec<String>),
    EmptyNamespace(String),
}

pub async fn resolve_scope(
    container: &Container,
    scope: &SearchScope,
) -> Result<ScopeResolution, DomainError> {
    Ok(match scope {
        SearchScope::All => ScopeResolution::All,
        SearchScope::Project(p) => ScopeResolution::Projects(vec![p.clone()]),
        SearchScope::Namespace(ns) => {
            let projects = container.node_repository()?.namespace_projects(ns).await?;
            if projects.is_empty() {
                ScopeResolution::EmptyNamespace(ns.clone())
            } else {
                ScopeResolution::Projects(projects)
            }
        }
    })
}

/// The result of a scoped search: the ranked hits, or a note that the requested
/// namespace has no member projects.
pub enum SearchOutcome {
    Hits(Vec<(MemoryItem, f32)>),
    EmptyNamespace(String),
}

/// Hybrid semantic + keyword search, scoped to all / a project / a namespace.
pub async fn search(
    container: &Container,
    query: &str,
    kind: Option<MemoryKind>,
    scope: &SearchScope,
    limit: usize,
) -> Result<SearchOutcome, DomainError> {
    let projects = match resolve_scope(container, scope).await? {
        ScopeResolution::All => None,
        ScopeResolution::Projects(p) => Some(p),
        ScopeResolution::EmptyNamespace(ns) => return Ok(SearchOutcome::EmptyNamespace(ns)),
    };
    let use_case: MemorySearchUseCase = container.memory_search_use_case()?;
    let hits = use_case
        .execute(query, kind, projects.as_deref(), limit)
        .await?;
    Ok(SearchOutcome::Hits(hits))
}

// ── Memory graph ─────────────────────────────────────────────────────────
//
// These are what every surface now calls. The item functions above are still
// compiled but no longer reachable from the CLI, HTTP or MCP layers; they go
// when the item layer is deleted.

/// The result of a scoped memory recall, mirroring [`SearchOutcome`].
pub enum MemorySearchOutcome {
    Hits(Vec<Recalled>),
    EmptyNamespace(String),
}

/// Ingest a transcript file into the memory graph.
pub async fn ingest_memories(
    container: &Container,
    path: &str,
    force: bool,
) -> Result<IngestionOutcome, DomainError> {
    let transcript = parse_transcript_file(std::path::Path::new(path))?;
    container
        .memory_ingestion_use_case()?
        .execute(&transcript, force)
        .await
}

/// Recall memories for `query` within `scope`.
pub async fn recall_memories(
    container: &Container,
    query: &str,
    kind: Option<MemoryKind>,
    scope: &SearchScope,
    limit: usize,
) -> Result<MemorySearchOutcome, DomainError> {
    let projects = match resolve_scope(container, scope).await? {
        ScopeResolution::All => None,
        ScopeResolution::Projects(p) => Some(p),
        ScopeResolution::EmptyNamespace(ns) => return Ok(MemorySearchOutcome::EmptyNamespace(ns)),
    };
    let hits = container
        .memory_recall_use_case()?
        .execute(query, kind, projects.as_deref(), limit)
        .await?;
    Ok(MemorySearchOutcome::Hits(hits))
}

/// Aggregate memory-store statistics.
pub async fn memory_stats(container: &Container) -> Result<MemoryStoreStats, DomainError> {
    container.memory_repository()?.memory_stats().await
}

/// List memories, newest first, optionally restricted by kind and lifecycle
/// status. `status` defaults to `active` at the call site, not here — a `None`
/// genuinely means "every status", which is what the conflict views want.
pub async fn list_memories(
    container: &Container,
    kind: Option<MemoryKind>,
    status: Option<MemoryStatus>,
) -> Result<Vec<Memory>, DomainError> {
    container
        .memory_repository()?
        .list_memories(kind, status, None)
        .await
}

/// What a `show <id>` against the memory store resolves to.
pub enum MemoryShowOutcome {
    /// A `memory://` node (the virtual filesystem survives the cutover).
    Node(MemoryNode),
    /// A memory plus its edges. An agent holding a memory id essentially always
    /// wants the neighbourhood too — what superseded it, what it refines — and
    /// making that a second round-trip just guarantees callers skip it.
    Memory {
        memory: Box<Memory>,
        edges: Vec<MemoryEdge>,
    },
    NotFound,
}

/// Resolve a reference: a `memory://` URI → a node; otherwise a memory id.
///
/// Unlike the item store there is no `kind/name` form, because a memory has no
/// name — its identity is its id, and two memories may state the same thing at
/// different times on purpose.
pub async fn show_memory(
    container: &Container,
    id: &str,
) -> Result<MemoryShowOutcome, DomainError> {
    if id.starts_with("memory://") {
        return Ok(match container.node_repository()?.find_node(id).await? {
            Some(node) => MemoryShowOutcome::Node(node),
            None => MemoryShowOutcome::NotFound,
        });
    }
    let repo = container.memory_repository()?;
    let Some(memory) = repo.find_memory(id).await? else {
        return Ok(MemoryShowOutcome::NotFound);
    };
    // Both directions: a memory's neighbourhood is what points at it as much as
    // what it points at.
    let mut edges = repo.edges_from(id).await?;
    edges.extend(repo.edges_to(id).await?);
    Ok(MemoryShowOutcome::Memory {
        memory: Box::new(memory),
        edges,
    })
}

/// What `forget <id>` did.
pub enum ForgetOutcome {
    Retracted,
    NotFound,
}

/// Retract a memory: flip it to `retracted` rather than deleting the row.
///
/// An append-only store has no delete, and `retracted` already carries the
/// exact meaning wanted here — "this was never true". No `retracts` *edge* is
/// written, despite the symmetry with the other transitions: an edge relates
/// two memories, and a manual forget has only one. Inventing a self-edge to fill
/// the slot would put a cycle in the graph to record something the status
/// already records.
pub async fn forget_memory(container: &Container, id: &str) -> Result<ForgetOutcome, DomainError> {
    let retracted = container
        .memory_repository()?
        .set_memory_status(id, MemoryStatus::Retracted, None)
        .await?;
    Ok(if retracted {
        ForgetOutcome::Retracted
    } else {
        ForgetOutcome::NotFound
    })
}

/// One unresolved disagreement: two memories that contradict each other and
/// are both still current.
pub struct Conflict {
    pub a: Memory,
    pub b: Memory,
    /// When the contradiction was recorded.
    pub recorded_at: i64,
}

/// The conflict queue, derived rather than stored.
///
/// A conflict is a `contradicts` edge whose two endpoints are both still
/// active. Once consolidation reconciles a pair — by writing a new memory that
/// supersedes them — at least one endpoint stops being active and the pair
/// drops out of this list on its own. Nothing has to remember to clear a flag,
/// and no memory can be stranded in a conflicted state that hides it from
/// recall: both sides keep answering queries the whole time.
pub async fn memory_conflicts(container: &Container) -> Result<Vec<Conflict>, DomainError> {
    let repo = container.memory_repository()?;
    let edges = repo.list_edges(Some(EdgeType::Contradicts)).await?;
    if edges.is_empty() {
        return Ok(Vec::new());
    }

    let mut ids: Vec<String> = Vec::with_capacity(edges.len() * 2);
    for edge in &edges {
        ids.push(edge.from_memory.clone());
        ids.push(edge.to_memory.clone());
    }
    ids.sort();
    ids.dedup();

    let by_id: std::collections::HashMap<String, Memory> = repo
        .find_memories(&ids)
        .await?
        .into_iter()
        .map(|m| (m.id.clone(), m))
        .collect();

    let mut conflicts = Vec::new();
    for edge in edges {
        let (Some(a), Some(b)) = (by_id.get(&edge.from_memory), by_id.get(&edge.to_memory)) else {
            continue;
        };
        // Either side no longer being current means the disagreement has been
        // settled — by consolidation, a later supersession, or a retraction.
        if a.status != MemoryStatus::Active || b.status != MemoryStatus::Active {
            continue;
        }
        conflicts.push(Conflict {
            a: a.clone(),
            b: b.clone(),
            recorded_at: edge.created_at,
        });
    }
    Ok(conflicts)
}

/// List stored items, optionally restricted to one kind, newest first.
pub async fn list_items(
    container: &Container,
    kind: Option<MemoryKind>,
) -> Result<Vec<MemoryItem>, DomainError> {
    container.node_repository()?.list_items(kind).await
}

/// What `show <id>` resolves to.
pub enum ShowOutcome {
    /// A `memory://` node.
    Node(MemoryNode),
    /// A single item (by id, or a uniquely-matching `kind/name`).
    Item(MemoryItem),
    /// A `kind/name` reference matching several items across projects.
    Many(Vec<MemoryItem>),
    /// Nothing matched the reference.
    NotFound,
}

/// Resolve a `show` reference: a `memory://` URI → a node; otherwise a
/// `kind/name` reference (which may match several projects) or an item id.
pub async fn show(container: &Container, id: &str) -> Result<ShowOutcome, DomainError> {
    let repo = container.node_repository()?;

    if id.starts_with("memory://") {
        return Ok(match repo.find_node(id).await? {
            Some(node) => ShowOutcome::Node(node),
            None => ShowOutcome::NotFound,
        });
    }

    if let Some((kind_str, name)) = id.split_once('/') {
        if let Some(kind) = MemoryKind::parse(kind_str) {
            let items = repo.find_items_named(kind, name).await?;
            match items.as_slice() {
                [item] => return Ok(ShowOutcome::Item(item.clone())),
                [] => {}
                _ => return Ok(ShowOutcome::Many(items)),
            }
        }
    }

    Ok(match repo.find_item_by_id(id).await? {
        Some(item) => ShowOutcome::Item(item),
        None => ShowOutcome::NotFound,
    })
}

/// What `delete <id>` did.
pub enum DeleteOutcome {
    /// One item was deleted.
    Deleted,
    /// A `kind/name` reference matched several items; nothing was deleted
    /// (deleting the wrong project's memory is unrecoverable).
    Ambiguous(Vec<MemoryItem>),
    /// Nothing matched.
    NotFound,
}

/// Delete by item id, or by a uniquely-matching `kind/name` reference. An
/// ambiguous `kind/name` (multiple projects) is refused, not guessed.
pub async fn delete(container: &Container, id: &str) -> Result<DeleteOutcome, DomainError> {
    let repo = container.node_repository()?;

    if let Some((kind_str, name)) = id.split_once('/') {
        if let Some(kind) = MemoryKind::parse(kind_str) {
            let items = repo.find_items_named(kind, name).await?;
            match items.as_slice() {
                [item] => {
                    repo.delete_item_by_id(item.id()).await?;
                    return Ok(DeleteOutcome::Deleted);
                }
                [] => {}
                _ => return Ok(DeleteOutcome::Ambiguous(items)),
            }
        }
    }

    Ok(if repo.delete_item_by_id(id).await? {
        DeleteOutcome::Deleted
    } else {
        DeleteOutcome::NotFound
    })
}

/// List imported sessions, newest first.
pub async fn sessions(container: &Container) -> Result<Vec<ImportedSession>, DomainError> {
    container.node_repository()?.list_sessions().await
}

/// Aggregate memory-store statistics.
pub async fn stats(container: &Container) -> Result<NodeStats, DomainError> {
    container.node_repository()?.stats().await
}

/// List the children of a virtual-filesystem directory. `uri = None` returns
/// the top-level roots (digest + sessions + resources).
pub async fn tree(
    container: &Container,
    uri: Option<&str>,
) -> Result<Vec<MemoryNode>, DomainError> {
    let repo = container.node_repository()?;
    Ok(match uri {
        None => {
            let mut nodes = Vec::new();
            if let Some(digest) = repo.find_node(MEMORY_ROOT_URI).await? {
                nodes.push(digest);
            }
            nodes.extend(repo.list_child_nodes(SESSIONS_ROOT_URI).await?);
            nodes.extend(repo.list_child_nodes(RESOURCES_ROOT_URI).await?);
            nodes
        }
        Some(dir) => repo.list_child_nodes(dir).await?,
    })
}

/// Import a transcript file, running extraction over it.
pub async fn import(
    container: &Container,
    path: &str,
    force: bool,
) -> Result<ImportOutcome, DomainError> {
    let transcript = parse_transcript_file(std::path::Path::new(path))?;
    container
        .memory_import_use_case()?
        .execute(&transcript, force)
        .await
}

/// A resource added to the memory virtual filesystem.
pub struct AddedResource {
    pub node: MemoryNode,
    pub source: String,
    pub chars: usize,
}

/// Fetch a file/URL, summarize it, and store it as a `memory://resources/…`
/// node. Best-effort digest refresh afterwards.
pub async fn add_resource(
    container: &Container,
    source: &str,
    name: Option<&str>,
) -> Result<AddedResource, DomainError> {
    // Fetch first — a bad path/URL should fail before we spin up the LLM.
    let fetched = fetch_resource(source)
        .await
        .map_err(|e| DomainError::internal(format!("failed to fetch resource '{source}': {e}")))?;
    let slug = resource_slug(name.unwrap_or(&fetched.title));

    let summary = container.memory_summary_use_case()?;
    let node = summary
        .summarize_resource(&slug, &fetched.source, &fetched.text)
        .await?;
    if let Err(e) = summary.regenerate_digest().await {
        tracing::warn!("failed to regenerate memory digest after `add`: {e}");
    }
    Ok(AddedResource {
        node,
        source: fetched.source,
        chars: fetched.text.len(),
    })
}

/// Run one dream cycle (harvest + consolidate). `idle_minutes` is the finished-
/// session inactivity window; a manual run always harvests.
pub async fn dream(container: &Container, idle_minutes: u64) -> Result<DreamReport, DomainError> {
    // Clamp instead of wrapping: an absurd idle window must not become a
    // negative threshold that makes still-active sessions eligible.
    let idle_secs = i64::try_from(idle_minutes.saturating_mul(60)).unwrap_or(i64::MAX);
    container
        .memory_dream_use_case()?
        .execute(idle_secs, true)
        .await
}

// ── Namespaces ───────────────────────────────────────────────────────────────

pub async fn create_namespace(container: &Container, name: &str) -> Result<bool, DomainError> {
    container.node_repository()?.create_namespace(name).await
}

pub async fn delete_namespace(container: &Container, name: &str) -> Result<bool, DomainError> {
    container.node_repository()?.delete_namespace(name).await
}

pub async fn assign_project(
    container: &Container,
    namespace: &str,
    project: &str,
) -> Result<bool, DomainError> {
    container
        .node_repository()?
        .assign_project(namespace, project)
        .await
}

pub async fn unassign_project(
    container: &Container,
    namespace: &str,
    project: &str,
) -> Result<bool, DomainError> {
    container
        .node_repository()?
        .unassign_project(namespace, project)
        .await
}

/// All namespaces with their member-project counts.
pub async fn list_namespaces(container: &Container) -> Result<Vec<(String, u64)>, DomainError> {
    container.node_repository()?.list_namespaces().await
}

/// A namespace's member projects.
pub async fn namespace_projects(
    container: &Container,
    name: &str,
) -> Result<Vec<String>, DomainError> {
    container.node_repository()?.namespace_projects(name).await
}

/// Whether a node's L2 content should be shown (Project digest nodes carry an
/// internal manifest that is bookkeeping, not user content).
pub fn node_has_visible_content(node: &MemoryNode) -> bool {
    node.kind() != NodeKind::Project && !node.content().trim().is_empty()
}
