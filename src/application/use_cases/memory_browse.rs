//! Unified memory recall for the TUI, as a navigable virtual filesystem.
//!
//! - **Browse** (empty query) returns the whole store as a flattened *tree* of
//!   [`MemoryRow`]s: the `memory://memory` digest (with its L0/L1 levels and the
//!   memory items grouped by category beneath it), then directory headers
//!   (`sessions/`, `resources/`) with each node under them, and — nested one
//!   level deeper — that node's L0/L1/L2 levels as their own selectable rows.
//!   Selecting a level row shows just that level on the right; selecting the
//!   node row shows its L0+L1 summary (the full L2 body is reached via the
//!   node's "L2 · detail" child row).
//! - **Search** (non-empty query) returns a flat, ranked list of rows (depth 0)
//!   from hybrid semantic + keyword recall over both items and nodes.
//!
//! The TUI renders the rows with indentation and drives a single flat cursor
//! over them, mirroring the call-context tree.

use std::collections::HashMap;
use std::sync::Arc;

use crate::application::interfaces::{Embedder, MemoryRepository, NodeRepository};
use crate::application::use_cases::memory_recall::MemoryRecallUseCase;
use crate::application::use_cases::memory_summary::{
    MEMORY_ROOT_URI, PROJECTS_ROOT_URI, RESOURCES_ROOT_URI, SESSIONS_ROOT_URI,
};
use crate::domain::{
    DomainError, EdgeType, EntityRef, Memory, MemoryKind, MemoryNode, MemoryStatus, NodeKind,
};

/// RRF dampening constant (matches [`MemorySearchUseCase`]).
const RRF_K: f32 = 60.0;

/// How many candidates the node legs retrieve before fusion.
const NODE_CANDIDATES_PER_LEG: usize = 20;

/// Sort rank for a node kind in the browse view, so the filesystem reads
/// top-down: the digest first, then project digests, sessions, resources.
fn node_kind_rank(kind: NodeKind) -> u8 {
    match kind {
        NodeKind::Memory => 0,
        NodeKind::Project => 1,
        NodeKind::Session => 2,
        NodeKind::Resource => 3,
    }
}

/// Which of a node's three levels a level row addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLevel {
    /// L0 — the one-line abstract.
    Abstract,
    /// L1 — the overview.
    Overview,
    /// L2 — the full detail (transcript / resource text).
    Detail,
}

impl MemoryLevel {
    pub fn tag(&self) -> &'static str {
        match self {
            MemoryLevel::Abstract => "L0 · abstract",
            MemoryLevel::Overview => "L1 · overview",
            MemoryLevel::Detail => "L2 · detail",
        }
    }
}

/// The payload a row points at — what the detail pane shows when it is selected.
#[derive(Debug, Clone)]
pub enum RowTarget {
    /// A directory header (`sessions/`, …) — not itself content.
    Directory,
    /// A collapsible group header in the grouped tree (`Memories`, a category
    /// like `Preferences`, `Projects`, `Sessions`). Carries a stable `key` the
    /// UI uses to track collapse state across refreshes, and the `count` of
    /// direct children shown as a badge.
    Group { key: String, count: usize },
    /// A whole node: the detail pane shows all its levels.
    Node(MemoryNode),
    /// A single level of a node: the detail pane shows just that level.
    NodeLevel {
        node: MemoryNode,
        level: MemoryLevel,
    },
    /// A flat memory item.
    Memory {
        memory: LabelledMemory,
        /// The memory's typed edges, already resolved to the other side's
        /// statement. Carried on the row rather than fetched when the detail
        /// pane opens: the TUI's selection handling is synchronous, and a
        /// personal store's whole edge table is a single small query per
        /// refresh — cheaper than plumbing a lazy async fetch through the
        /// screen for a handful of rows.
        links: Vec<MemoryLink>,
    },
}

/// A memory plus its display labels, so a row can show "orders-events" rather
/// than the entity UUID the memory actually stores.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelledMemory {
    pub memory: Memory,
    pub subject: Option<String>,
    pub object: Option<String>,
}

impl LabelledMemory {
    /// The row title: the whole triple, `subject · predicate · object`.
    ///
    /// Not the statement — a self-contained sentence repeated down a list reads
    /// as a wall of prose. Not subject+predicate either: without the object,
    /// two memories about one subject render identically and the row looks
    /// truncated rather than deliberately short.
    pub fn title(&self) -> String {
        let Some(subject) = &self.subject else {
            return self.memory.statement.clone();
        };
        let predicate = self.memory.predicate.as_str();
        match &self.object {
            Some(object) => format!("{subject} · {predicate} · {object}"),
            None => format!("{subject} · {predicate}"),
        }
    }
}

/// One edge of a memory, resolved for display.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryLink {
    pub edge_type: EdgeType,
    /// `true` when this memory is the edge's source. Direction is shown rather
    /// than normalised away: "supersedes X" and "superseded by X" are opposite
    /// facts about the memory you are looking at.
    pub outgoing: bool,
    pub other_id: String,
    /// The other side's statement, or its id when that memory is missing.
    pub other_statement: String,
}

#[derive(Debug, Clone)]
pub struct MemoryRow {
    /// Indentation depth (0 = top level).
    pub depth: u8,
    /// Kind label shown in the row (`session`, `resource`, `preference`, …), or
    /// empty for level rows / directories.
    pub kind_label: String,
    /// Primary text of the row (a URI, an item name, a level tag, a dir name).
    pub label: String,
    /// One-line preview shown under the label (abstracts / content snippets).
    pub preview: Option<String>,
    /// Relevance score, `Some` only for search-result rows.
    pub score: Option<f32>,
    /// What selecting this row shows in the detail pane.
    pub target: RowTarget,
}

/// Combined search/browse over memory items and virtual-filesystem nodes.
pub struct MemoryBrowseUseCase {
    node_repo: Arc<dyn NodeRepository>,
    memory_repo: Arc<dyn MemoryRepository>,
    embedder: Embedder,
    recall: MemoryRecallUseCase,
}

impl MemoryBrowseUseCase {
    pub fn new(
        node_repo: Arc<dyn NodeRepository>,
        memory_repo: Arc<dyn MemoryRepository>,
        embedder: Embedder,
    ) -> Self {
        let recall = MemoryRecallUseCase::new(Arc::clone(&memory_repo), embedder.clone());
        Self {
            node_repo,
            memory_repo,
            embedder,
            recall,
        }
    }

    /// Canonical names for every entity these memories reference, in one query.
    async fn entity_labels(
        &self,
        memories: &[Memory],
    ) -> Result<HashMap<String, String>, DomainError> {
        let mut ids: Vec<String> = memories
            .iter()
            .flat_map(|m| [m.subject.entity_id(), m.object.entity_id()])
            .flatten()
            .map(str::to_string)
            .collect();
        ids.sort();
        ids.dedup();
        if ids.is_empty() {
            return Ok(HashMap::new());
        }
        Ok(self
            .memory_repo
            .find_entities(&ids)
            .await?
            .into_iter()
            .map(|e| (e.id, e.canonical_name))
            .collect())
    }

    /// Resolve every edge touching `memories` into display-ready links.
    ///
    /// Two queries regardless of how many memories there are: one for the
    /// edges, one to resolve the statements on the far side (which may be
    /// superseded, and so absent from the caller's list).
    async fn link_index(
        &self,
        memories: &[Memory],
    ) -> Result<HashMap<String, Vec<MemoryLink>>, DomainError> {
        let mut index: HashMap<String, Vec<MemoryLink>> = HashMap::new();
        if memories.is_empty() {
            return Ok(index);
        }
        let ids: Vec<String> = memories.iter().map(|m| m.id.clone()).collect();
        let edges = self.memory_repo.edges_for(&ids).await?;
        if edges.is_empty() {
            return Ok(index);
        }

        let mut referenced: Vec<String> = edges
            .iter()
            .flat_map(|e| [e.from_memory.clone(), e.to_memory.clone()])
            .collect();
        referenced.sort();
        referenced.dedup();
        let statements: HashMap<String, String> = self
            .memory_repo
            .find_memories(&referenced)
            .await?
            .into_iter()
            .map(|m| (m.id, m.statement))
            .collect();
        let label = |id: &str| {
            statements
                .get(id)
                .cloned()
                .unwrap_or_else(|| id.to_string())
        };

        let owned: std::collections::HashSet<&String> = ids.iter().collect();
        for edge in &edges {
            // Both endpoints are considered: an edge between two listed
            // memories is a fact about each of them.
            if owned.contains(&edge.from_memory) {
                index
                    .entry(edge.from_memory.clone())
                    .or_default()
                    .push(MemoryLink {
                        edge_type: edge.edge_type,
                        outgoing: true,
                        other_id: edge.to_memory.clone(),
                        other_statement: label(&edge.to_memory),
                    });
            }
            if owned.contains(&edge.to_memory) {
                index
                    .entry(edge.to_memory.clone())
                    .or_default()
                    .push(MemoryLink {
                        edge_type: edge.edge_type,
                        outgoing: false,
                        other_id: edge.from_memory.clone(),
                        other_statement: label(&edge.from_memory),
                    });
            }
        }
        Ok(index)
    }

    /// Produce the rows to display: the filesystem tree when `query` is empty,
    /// a ranked flat list of hits otherwise.
    pub async fn execute(&self, query: &str, limit: usize) -> Result<Vec<MemoryRow>, DomainError> {
        let query = query.trim();
        if query.is_empty() {
            self.browse_tree().await
        } else {
            self.search(query, limit).await
        }
    }

    /// The memory store as a *grouped* tree for the TUI: category groups with
    /// counts, then leaves — and no inline L0/L1/L2 level rows (the detail pane
    /// shows those). When `query` is non-empty this defers to the same ranked
    /// hit list as [`Self::execute`], so a search replaces the tree with results.
    ///
    /// Layout (each group header is a collapsible [`RowTarget::Group`]):
    /// ```text
    /// Memories (9)
    ///   Preferences (1)
    ///     memory_view_default
    ///   Skills (2)
    ///     …
    ///   Facts (6)
    ///     …
    /// Projects (1)
    ///   github.com/matter-js/matter.js
    /// Sessions (130)
    ///   <session title>
    /// ```
    /// Empty groups are omitted. Items keep their `(kind, name, project)`
    /// identity; only the presentation is grouped.
    pub async fn grouped_tree(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemoryRow>, DomainError> {
        let query = query.trim();
        if !query.is_empty() {
            return self.search(query, limit).await;
        }

        let memories = self
            .memory_repo
            .list_memories(None, Some(MemoryStatus::Active), None)
            .await?;
        let links = self.link_index(&memories).await?;
        let labels = self.entity_labels(&memories).await?;
        let nodes = self.node_repo.list_nodes(None).await?;

        let mut rows: Vec<MemoryRow> = Vec::new();

        // ── Memories: one group over all of them, with a category subgroup per
        //    non-empty kind (preferences/experiences/skills/facts). ──────────
        if !memories.is_empty() {
            rows.push(group_row("memories", "Memories", memories.len(), 0));
            for kind in MemoryKind::ALL {
                let group: Vec<&Memory> = memories.iter().filter(|m| m.kind == kind).collect();
                if group.is_empty() {
                    continue;
                }
                rows.push(group_row(
                    &format!("memories/{}", kind.as_str()),
                    kind.plural_title(),
                    group.len(),
                    1,
                ));
                for memory in group {
                    rows.push(memory_row(memory, &links, &labels, 2, None));
                }
            }
        }

        // ── Projects and Sessions: a group per node kind, node leaves under it
        //    (no level rows). Resources join Projects-style under their own
        //    group so nothing is orphaned. ────────────────────────────────────
        push_node_group(&mut rows, "projects", "Projects", NodeKind::Project, &nodes);
        push_node_group(&mut rows, "sessions", "Sessions", NodeKind::Session, &nodes);
        push_node_group(
            &mut rows,
            "resources",
            "Resources",
            NodeKind::Resource,
            &nodes,
        );

        Ok(rows)
    }

    /// Hybrid semantic + keyword recall over items *and* nodes, fused per
    /// modality and interleaved by score into a flat list of depth-0 rows.
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryRow>, DomainError> {
        let hits = self.recall.execute(query, None, None, limit).await?;
        let nodes = self.search_nodes(query, limit).await?;

        // Only the hits need links, so the index is built over them rather than
        // the whole store.
        let found: Vec<Memory> = hits.iter().map(|h| h.memory.clone()).collect();
        let links = self.link_index(&found).await?;
        let labels = self.entity_labels(&found).await?;

        let mut scored: Vec<(f32, MemoryRow)> = Vec::new();
        for hit in hits {
            let score = hit.score;
            scored.push((
                score,
                memory_row(&hit.memory, &links, &labels, 0, Some(score)),
            ));
        }
        for (node, score) in nodes {
            scored.push((score, node_row(&node, 0, Some(score))));
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored.into_iter().map(|(_, row)| row).collect())
    }

    /// Hybrid semantic + keyword recall over nodes, fused with RRF.
    async fn search_nodes(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(MemoryNode, f32)>, DomainError> {
        let semantic = if self.embedder.embeddings_enabled() {
            let vector = self.embedder.embed_query(query).await?;
            self.node_repo
                .search_nodes_semantic(&vector, None, NODE_CANDIDATES_PER_LEG)
                .await?
        } else {
            Vec::new()
        };
        let keyword = self
            .node_repo
            .search_nodes_keyword(query, None, NODE_CANDIDATES_PER_LEG)
            .await?;

        let mut fused: HashMap<String, (MemoryNode, f32)> = HashMap::new();
        for results in [semantic, keyword] {
            for (rank, (node, _score)) in results.into_iter().enumerate() {
                let contribution = 1.0 / (RRF_K + rank as f32 + 1.0);
                fused
                    .entry(node.uri().to_string())
                    .and_modify(|(_, score)| *score += contribution)
                    .or_insert((node, contribution));
            }
        }
        let mut results: Vec<(MemoryNode, f32)> = fused.into_values().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        Ok(results)
    }

    /// Browse the whole virtual filesystem as a flattened tree.
    ///
    /// Layout (always fully expanded):
    /// ```text
    /// memory://memory            (digest node)
    ///   L0 · abstract
    ///   L1 · overview
    ///   preferences/             (item categories nest under the digest)
    ///     [preference] commit_style
    ///   facts/
    ///     [fact] duckdb_locks
    /// sessions/                  (directory)
    ///   memory://sessions/<id>
    ///     L0 · abstract
    ///     L1 · overview
    ///     L2 · detail
    /// resources/                 (directory)
    ///   memory://resources/<slug>
    ///     L0 · abstract  …
    /// ```
    /// Before the first digest exists, items fall back to a top-level
    /// `memory/` directory so they are never orphaned.
    async fn browse_tree(&self) -> Result<Vec<MemoryRow>, DomainError> {
        let memories = self
            .memory_repo
            .list_memories(None, Some(MemoryStatus::Active), None)
            .await?;
        let links = self.link_index(&memories).await?;
        let labels = self.entity_labels(&memories).await?;
        let mut nodes = self.node_repo.list_nodes(None).await?;

        nodes.sort_by(|a, b| {
            node_kind_rank(a.kind())
                .cmp(&node_kind_rank(b.kind()))
                .then_with(|| (b.uri() == MEMORY_ROOT_URI).cmp(&(a.uri() == MEMORY_ROOT_URI)))
                .then_with(|| b.updated_at().cmp(&a.updated_at()))
        });

        let mut rows: Vec<MemoryRow> = Vec::new();

        // The digest sits at the filesystem root (depth 0), with its levels
        // (L0/L1) and the grouped memory items nested directly beneath it, so
        // everything durable lives under one `memory` root.
        let digest: Vec<&MemoryNode> = nodes
            .iter()
            .filter(|n| n.kind() == NodeKind::Memory)
            .collect();
        let has_digest = !digest.is_empty();
        for node in digest {
            push_node_with_levels(&mut rows, node, 0);
        }

        // Items grouped by kind: one sub-directory per category
        // (preferences/experiences/skills/facts), each holding its items, empty
        // categories omitted. Nest them under the digest (depth 1/2) when it
        // exists; otherwise fall back to a top-level `memory/` dir so items are
        // never orphaned before the first digest is generated.
        if !memories.is_empty() {
            let base_depth = if has_digest {
                1
            } else {
                rows.push(dir_row("memory/", 0));
                1
            };
            push_memory_groups(&mut rows, &memories, &links, &labels, base_depth);
        }

        // Project digests, sessions, and resources each get a directory header,
        // with their nodes (and each node's levels) nested underneath.
        push_dir_group(
            &mut rows,
            "projects/",
            PROJECTS_ROOT_URI,
            NodeKind::Project,
            &nodes,
        );
        push_dir_group(
            &mut rows,
            "sessions/",
            SESSIONS_ROOT_URI,
            NodeKind::Session,
            &nodes,
        );
        push_dir_group(
            &mut rows,
            "resources/",
            RESOURCES_ROOT_URI,
            NodeKind::Resource,
            &nodes,
        );

        Ok(rows)
    }
}

/// Append one category sub-directory per non-empty memory kind (at
/// `category_depth`) with its items nested one level deeper.
fn push_memory_groups(
    rows: &mut Vec<MemoryRow>,
    memories: &[Memory],
    links: &HashMap<String, Vec<MemoryLink>>,
    labels: &HashMap<String, String>,
    category_depth: u8,
) {
    for kind in MemoryKind::ALL {
        let group: Vec<&Memory> = memories.iter().filter(|m| m.kind == kind).collect();
        if group.is_empty() {
            continue;
        }
        rows.push(dir_row(&format!("{}/", kind.plural()), category_depth));
        for memory in group {
            rows.push(memory_row(memory, links, labels, category_depth + 1, None));
        }
    }
}

/// Append a directory header row plus each node of `kind` (with its levels).
fn push_dir_group(
    rows: &mut Vec<MemoryRow>,
    dir_label: &str,
    _dir_uri: &str,
    kind: NodeKind,
    nodes: &[MemoryNode],
) {
    let group: Vec<&MemoryNode> = nodes.iter().filter(|n| n.kind() == kind).collect();
    if group.is_empty() {
        return;
    }
    rows.push(dir_row(dir_label, 0));
    for node in group {
        push_node_with_levels(rows, node, 1);
    }
}

/// Append a node row followed by one child row per present level.
fn push_node_with_levels(rows: &mut Vec<MemoryRow>, node: &MemoryNode, depth: u8) {
    rows.push(node_row(node, depth, None));
    let child_depth = depth + 1;
    // L0 always exists.
    rows.push(level_row(node, MemoryLevel::Abstract, child_depth));
    if !node.overview().trim().is_empty() {
        rows.push(level_row(node, MemoryLevel::Overview, child_depth));
    }
    // Mask internal manifest for Project digest nodes (index nodes have
    // empty content by invariant; the manifest is bookkeeping).
    let has_content = if node.kind() == NodeKind::Project {
        false
    } else {
        !node.content().trim().is_empty()
    };
    if has_content {
        rows.push(level_row(node, MemoryLevel::Detail, child_depth));
    }
}

fn dir_row(label: &str, depth: u8) -> MemoryRow {
    MemoryRow {
        depth,
        kind_label: String::new(),
        label: label.to_string(),
        preview: None,
        score: None,
        target: RowTarget::Directory,
    }
}

/// A collapsible group header row for the grouped tree.
fn group_row(key: &str, label: &str, count: usize, depth: u8) -> MemoryRow {
    MemoryRow {
        depth,
        kind_label: String::new(),
        label: label.to_string(),
        preview: None,
        score: None,
        target: RowTarget::Group {
            key: key.to_string(),
            count,
        },
    }
}

/// Append a group header for `kind`'s nodes (with count) plus one leaf row per
/// node — no L0/L1/L2 level rows. Omitted entirely when the group is empty.
fn push_node_group(
    rows: &mut Vec<MemoryRow>,
    key: &str,
    label: &str,
    kind: NodeKind,
    nodes: &[MemoryNode],
) {
    let group: Vec<&MemoryNode> = nodes.iter().filter(|n| n.kind() == kind).collect();
    if group.is_empty() {
        return;
    }
    rows.push(group_row(key, label, group.len(), 0));
    for node in group {
        rows.push(node_row(node, 1, None));
    }
}

fn node_row(node: &MemoryNode, depth: u8, score: Option<f32>) -> MemoryRow {
    MemoryRow {
        depth,
        kind_label: node.kind().to_string(),
        label: node.uri().to_string(),
        preview: one_line(node.abstract_()),
        score,
        target: RowTarget::Node(node.clone()),
    }
}

fn level_row(node: &MemoryNode, level: MemoryLevel, depth: u8) -> MemoryRow {
    let text = match level {
        MemoryLevel::Abstract => node.abstract_(),
        MemoryLevel::Overview => node.overview(),
        MemoryLevel::Detail => {
            // Mask internal manifest for Project digest nodes (index nodes have
            // empty content by invariant; the manifest is bookkeeping).
            if node.kind() == NodeKind::Project {
                ""
            } else {
                node.content()
            }
        }
    };
    MemoryRow {
        depth,
        kind_label: String::new(),
        label: level.tag().to_string(),
        preview: one_line(text),
        score: None,
        target: RowTarget::NodeLevel {
            node: node.clone(),
            level,
        },
    }
}

fn memory_row(
    memory: &Memory,
    links: &HashMap<String, Vec<MemoryLink>>,
    labels: &HashMap<String, String>,
    depth: u8,
    score: Option<f32>,
) -> MemoryRow {
    let labelled = LabelledMemory {
        memory: memory.clone(),
        subject: entity_label(&memory.subject, labels),
        object: entity_label(&memory.object, labels),
    };
    MemoryRow {
        depth,
        kind_label: memory.kind.as_str().to_string(),
        // Short title in the row; the full statement is the preview and the
        // detail pane. A list of self-contained sentences is unreadable.
        label: labelled.title(),
        preview: one_line(&memory.statement),
        score,
        target: RowTarget::Memory {
            memory: labelled,
            links: links.get(&memory.id).cloned().unwrap_or_default(),
        },
    }
}

/// An entity ref rendered for display: the entity's canonical name when it
/// resolves, the literal value otherwise.
fn entity_label(r: &EntityRef, labels: &HashMap<String, String>) -> Option<String> {
    match r {
        EntityRef::Entity(id) => Some(labels.get(id).cloned().unwrap_or_else(|| id.clone())),
        EntityRef::Literal(v) if v.trim().is_empty() => None,
        EntityRef::Literal(v) => Some(v.clone()),
    }
}

/// Collapse whitespace to a single-line preview, or `None` when empty.
fn one_line(text: &str) -> Option<String> {
    let s: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Predicate;

    fn node(uri: &str, kind: NodeKind, overview: &str, content: &str) -> MemoryNode {
        MemoryNode::new(
            uri.into(),
            kind,
            None,
            "an abstract".into(),
            overview.into(),
            content.into(),
            0,
            0,
        )
    }

    #[test]
    fn push_node_with_levels_emits_present_levels_only() {
        let mut rows = Vec::new();
        // Only L0 present (no overview, no content).
        push_node_with_levels(
            &mut rows,
            &node("memory://x", NodeKind::Resource, "", ""),
            1,
        );
        assert_eq!(rows.len(), 2); // node + L0
        assert!(matches!(rows[0].target, RowTarget::Node(_)));
        assert!(matches!(
            rows[1].target,
            RowTarget::NodeLevel {
                level: MemoryLevel::Abstract,
                ..
            }
        ));

        // All three levels present.
        let mut rows = Vec::new();
        push_node_with_levels(
            &mut rows,
            &node("memory://y", NodeKind::Session, "ov", "detail"),
            1,
        );
        assert_eq!(rows.len(), 4); // node + L0 + L1 + L2
        assert_eq!(rows[1].label, "L0 · abstract");
        assert_eq!(rows[2].label, "L1 · overview");
        assert_eq!(rows[3].label, "L2 · detail");
        // Child rows are nested one level deeper than the node row.
        assert_eq!(rows[0].depth, 1);
        assert_eq!(rows[1].depth, 2);
    }

    fn memory(kind: MemoryKind, statement: &str) -> Memory {
        Memory {
            id: statement.into(),
            kind,
            subject: crate::domain::EntityRef::Literal("the user".into()),
            predicate: Predicate::Prefers,
            object: crate::domain::EntityRef::Literal("tabs".into()),
            statement: statement.into(),
            project: None,
            recorded_at: 1,
            valid_from: 1,
            valid_to: None,
            source_session_id: None,
            source_message_index: None,
            source_kind: crate::domain::SourceKind::UserStated,
            confidence: 0.9,
            status: MemoryStatus::Active,
            derived: false,
            derived_from: Vec::new(),
        }
    }

    #[test]
    fn push_memory_groups_nests_memories_by_category() {
        let memories = vec![
            memory(MemoryKind::Fact, "duckdb takes a file lock"),
            memory(MemoryKind::Preference, "the user prefers short commits"),
            memory(MemoryKind::Fact, "the storage engine is columnar"),
        ];
        let mut rows = Vec::new();
        // Category dirs at depth 1, memories at depth 2 (as when nested under
        // the digest).
        push_memory_groups(&mut rows, &memories, &HashMap::new(), &HashMap::new(), 1);

        // Categories follow MemoryKind::ALL order (preferences before facts);
        // the empty experience/skill kinds are omitted entirely.
        let dirs: Vec<&str> = rows
            .iter()
            .filter(|r| matches!(r.target, RowTarget::Directory))
            .map(|r| r.label.as_str())
            .collect();
        assert_eq!(dirs, vec!["preferences/", "facts/"]);

        assert!(rows.iter().all(|r| match r.target {
            RowTarget::Directory => r.depth == 1,
            RowTarget::Memory { .. } => r.depth == 2,
            _ => false,
        }));
        let leaves = rows
            .iter()
            .filter(|r| matches!(r.target, RowTarget::Memory { .. }))
            .count();
        assert_eq!(leaves, 3);
    }

    /// The row label is the statement, because that is the only part of a
    /// memory a human reads — a memory has no name to fall back on.
    #[test]
    fn a_memory_row_is_labelled_by_its_statement() {
        let m = memory(MemoryKind::Fact, "duckdb takes a file lock");
        // No entity labels resolved, and the subject is a literal, so the row
        // falls back to the subject text plus predicate rather than the whole
        // statement.
        let row = memory_row(&m, &HashMap::new(), &HashMap::new(), 0, None);
        assert_eq!(row.label, "the user · prefers · tabs");
        assert_eq!(row.preview.as_deref(), Some("duckdb takes a file lock"));
        assert_eq!(row.kind_label, "fact");
    }

    use crate::connector::adapter::DuckdbStore;
    use std::sync::Arc;

    #[tokio::test]
    async fn grouped_tree_matches_the_app_shape() {
        let repo = Arc::new(DuckdbStore::in_memory(4, "mock").unwrap());
        // Two facts, one preference, and a session node. Seeded through the
        // *memory* port — the projection the import path actually writes, so
        // this test fails if the tree is ever pointed back at the item table.
        for m in [
            memory(MemoryKind::Fact, "duckdb takes a file lock"),
            memory(MemoryKind::Fact, "the storage engine is columnar"),
            memory(MemoryKind::Preference, "the user prefers short commits"),
        ] {
            crate::application::MemoryRepository::append_memory(repo.as_ref(), &m, None)
                .await
                .unwrap();
        }
        repo.upsert_node(
            &node(
                "memory://sessions/abc",
                NodeKind::Session,
                "ov",
                "transcript",
            ),
            None,
        )
        .await
        .unwrap();

        let use_case = MemoryBrowseUseCase::new(repo.clone(), repo.clone(), Embedder::disabled());
        let rows = use_case.grouped_tree("", 50).await.unwrap();

        // Group headers, in order, with their counts.
        let groups: Vec<(String, usize)> = rows
            .iter()
            .filter_map(|r| match &r.target {
                RowTarget::Group { count, .. } => Some((r.label.clone(), *count)),
                _ => None,
            })
            .collect();
        assert_eq!(
            groups,
            vec![
                ("Memories".to_string(), 3),
                ("Preferences".to_string(), 1),
                ("Facts".to_string(), 2),
                ("Sessions".to_string(), 1),
            ]
        );

        // No inline L0/L1/L2 level rows in the grouped tree.
        assert!(
            !rows
                .iter()
                .any(|r| matches!(r.target, RowTarget::NodeLevel { .. })),
            "grouped tree must not carry level rows"
        );

        // Category subgroups are nested (depth 1) under Memories (depth 0);
        // items are depth 2; the Sessions group is a top-level (depth 0) header.
        let memories = &rows[0];
        assert_eq!(memories.depth, 0);
        assert!(matches!(memories.target, RowTarget::Group { .. }));
        let prefs = rows
            .iter()
            .find(|r| r.label == "Preferences")
            .expect("preferences group");
        assert_eq!(prefs.depth, 1);
    }

    #[tokio::test]
    async fn grouped_tree_defers_to_search_when_querying() {
        let repo = Arc::new(DuckdbStore::in_memory(4, "mock").unwrap());
        crate::application::MemoryRepository::append_memory(
            repo.as_ref(),
            &memory(MemoryKind::Fact, "retry network timeouts with backoff"),
            None,
        )
        .await
        .unwrap();
        let use_case = MemoryBrowseUseCase::new(repo.clone(), repo.clone(), Embedder::disabled());

        // A non-empty query returns ranked hit rows (depth 0), not group headers.
        let rows = use_case.grouped_tree("network", 10).await.unwrap();
        assert!(!rows.is_empty());
        assert!(
            rows.iter()
                .all(|r| !matches!(r.target, RowTarget::Group { .. })),
            "search results should not contain group headers"
        );
    }
}
