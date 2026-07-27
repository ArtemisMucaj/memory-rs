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

use crate::application::interfaces::{Embedder, MemoryRepository};
use crate::application::use_cases::memory_search::MemorySearchUseCase;
use crate::application::use_cases::memory_summary::{
    MEMORY_ROOT_URI, PROJECTS_ROOT_URI, RESOURCES_ROOT_URI, SESSIONS_ROOT_URI,
};
use crate::domain::{DomainError, MemoryItem, MemoryKind, MemoryNode, NodeKind};

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
    Item(MemoryItem),
}

/// One rendered row in the memory tree/list.
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
    memory_repo: Arc<dyn MemoryRepository>,
    embedder: Embedder,
    item_search: MemorySearchUseCase,
}

impl MemoryBrowseUseCase {
    pub fn new(memory_repo: Arc<dyn MemoryRepository>, embedder: Embedder) -> Self {
        let embedder_arc = Arc::new(embedder.clone());
        let item_search =
            MemorySearchUseCase::new(Arc::clone(&memory_repo), Arc::clone(&embedder_arc));
        Self {
            memory_repo,
            embedder,
            item_search,
        }
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

        let items = self.memory_repo.list_items(None).await?;
        let nodes = self.memory_repo.list_nodes(None).await?;

        let mut rows: Vec<MemoryRow> = Vec::new();

        // ── Memories: one group over all items, with a category subgroup per
        //    non-empty kind (preferences/experiences/skills/facts). ──────────
        if !items.is_empty() {
            rows.push(group_row("memories", "Memories", items.len(), 0));
            for kind in MemoryKind::ALL {
                let group: Vec<&MemoryItem> = items.iter().filter(|i| i.kind() == kind).collect();
                if group.is_empty() {
                    continue;
                }
                rows.push(group_row(
                    &format!("memories/{}", kind.as_str()),
                    kind.plural_title(),
                    group.len(),
                    1,
                ));
                for item in group {
                    rows.push(item_row(item, 2, None));
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
        let items = self.item_search.execute(query, None, None, limit).await?;
        let nodes = self.search_nodes(query, limit).await?;

        let mut scored: Vec<(f32, MemoryRow)> = Vec::new();
        for (item, score) in items {
            scored.push((score, item_row(&item, 0, Some(score))));
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
            self.memory_repo
                .search_nodes_semantic(&vector, None, NODE_CANDIDATES_PER_LEG)
                .await?
        } else {
            Vec::new()
        };
        let keyword = self
            .memory_repo
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
        let items = self.memory_repo.list_items(None).await?;
        let mut nodes = self.memory_repo.list_nodes(None).await?;

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
        if !items.is_empty() {
            let base_depth = if has_digest {
                1
            } else {
                rows.push(dir_row("memory/", 0));
                1
            };
            push_item_groups(&mut rows, &items, base_depth);
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
fn push_item_groups(rows: &mut Vec<MemoryRow>, items: &[MemoryItem], category_depth: u8) {
    for kind in MemoryKind::ALL {
        let group: Vec<&MemoryItem> = items.iter().filter(|i| i.kind() == kind).collect();
        if group.is_empty() {
            continue;
        }
        rows.push(dir_row(&format!("{}/", kind.plural()), category_depth));
        for item in group {
            rows.push(item_row(item, category_depth + 1, None));
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

fn item_row(item: &MemoryItem, depth: u8, score: Option<f32>) -> MemoryRow {
    MemoryRow {
        depth,
        kind_label: item.kind().to_string(),
        label: item.name().to_string(),
        preview: one_line(item.content()),
        score,
        target: RowTarget::Item(item.clone()),
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

    fn item(kind: MemoryKind, name: &str) -> MemoryItem {
        MemoryItem::new(
            name.into(),
            kind,
            name.into(),
            "content".into(),
            None,
            None,
            0,
            0,
            0,
        )
    }

    #[test]
    fn push_item_groups_nests_items_by_category() {
        let items = vec![
            item(MemoryKind::Fact, "duckdb_locks"),
            item(MemoryKind::Preference, "commit_style"),
            item(MemoryKind::Fact, "storage_engine"),
        ];
        let mut rows = Vec::new();
        // Category dirs at depth 1, items at depth 2 (as when nested under the
        // digest).
        push_item_groups(&mut rows, &items, 1);

        // Categories follow MemoryKind::ALL order (preferences before facts);
        // the empty experience/skill kinds are omitted entirely.
        let dirs: Vec<&str> = rows
            .iter()
            .filter(|r| matches!(r.target, RowTarget::Directory))
            .map(|r| r.label.as_str())
            .collect();
        assert_eq!(dirs, vec!["preferences/", "facts/"]);

        // Every dir is at depth 1 and every item at depth 2, and both facts are
        // grouped under the single `facts/` header.
        assert!(rows.iter().all(|r| match r.target {
            RowTarget::Directory => r.depth == 1,
            RowTarget::Item(_) => r.depth == 2,
            _ => false,
        }));
        let item_count = rows
            .iter()
            .filter(|r| matches!(r.target, RowTarget::Item(_)))
            .count();
        assert_eq!(item_count, 3);
    }

    use crate::connector::adapter::DuckdbMemoryRepository;
    use std::sync::Arc;

    #[tokio::test]
    async fn grouped_tree_matches_the_app_shape() {
        let repo = Arc::new(DuckdbMemoryRepository::in_memory(4, "mock").unwrap());
        // Two facts, one preference, and a session node.
        for it in [
            item(MemoryKind::Fact, "duckdb_locks"),
            item(MemoryKind::Fact, "storage_engine"),
            item(MemoryKind::Preference, "commit_style"),
        ] {
            repo.upsert_item(&it, None).await.unwrap();
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

        let use_case = MemoryBrowseUseCase::new(repo, Embedder::disabled());
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
        let repo = Arc::new(DuckdbMemoryRepository::in_memory(4, "mock").unwrap());
        repo.upsert_item(&item(MemoryKind::Fact, "network_timeout"), None)
            .await
            .unwrap();
        let use_case = MemoryBrowseUseCase::new(repo, Embedder::disabled());

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
