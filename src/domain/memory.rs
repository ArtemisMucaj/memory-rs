use serde::{Deserialize, Serialize};

/// Category of a long-term memory item extracted from a session.
///
/// The taxonomy is reduced to the kinds that matter for a coding assistant:
///
/// - `Preference` — what the user likes/dislikes or is accustomed to
///   (code style, communication style, tooling, workflow).
/// - `Experience` — a generalizable, reusable insight distilled from a
///   session: what situation triggers it, what approach works, and why.
/// - `Skill` — reusable procedural knowledge: a repeatable flow that could
///   become an automated skill (steps, prerequisites, failure modes).
/// - `Fact` — durable declarative information worth remembering (project
///   facts, environment details, decisions and their rationale).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryKind {
    Preference,
    Experience,
    Skill,
    Fact,
}

impl MemoryKind {
    pub const ALL: [MemoryKind; 4] = [
        MemoryKind::Preference,
        MemoryKind::Experience,
        MemoryKind::Skill,
        MemoryKind::Fact,
    ];

    /// Stable identifier used in storage and in the extraction JSON protocol.
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryKind::Preference => "preference",
            MemoryKind::Experience => "experience",
            MemoryKind::Skill => "skill",
            MemoryKind::Fact => "fact",
        }
    }

    /// Plural field name used in the extraction output JSON.
    pub fn plural(&self) -> &'static str {
        match self {
            MemoryKind::Preference => "preferences",
            MemoryKind::Experience => "experiences",
            MemoryKind::Skill => "skills",
            MemoryKind::Fact => "facts",
        }
    }

    /// Title-cased plural, used for group headers in the memory tree.
    pub fn plural_title(&self) -> &'static str {
        match self {
            MemoryKind::Preference => "Preferences",
            MemoryKind::Experience => "Experiences",
            MemoryKind::Skill => "Skills",
            MemoryKind::Fact => "Facts",
        }
    }

    pub fn parse(s: &str) -> Option<MemoryKind> {
        match s.trim().to_ascii_lowercase().as_str() {
            "preference" | "preferences" => Some(MemoryKind::Preference),
            "experience" | "experiences" => Some(MemoryKind::Experience),
            "skill" | "skills" => Some(MemoryKind::Skill),
            "fact" | "facts" => Some(MemoryKind::Fact),
            _ => None,
        }
    }
}

impl std::fmt::Display for MemoryKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A single long-term memory item.
///
/// Items are unique per `(kind, name)`: re-extracting the same topic updates
/// the existing item (content is rewritten by the extraction model with the
/// previous content in context) rather than creating a duplicate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    id: String,
    kind: MemoryKind,
    /// Short snake_case identifier for the memory topic
    /// (e.g. `rust_error_handling_style`, `duckdb_lock_conflict_fix`).
    name: String,
    /// Markdown content of the memory.
    content: String,
    /// Identifier of the session this memory was last extracted from.
    source_session_id: Option<String>,
    /// Project this memory belongs to (e.g. a repository directory name), or
    /// `None` when it applies globally across all projects. Project-specific
    /// insights (a fix for one codebase's SDK, a repo's build quirk) carry a
    /// project so they don't surface as advice in unrelated projects.
    project: Option<String>,
    created_at: i64,
    updated_at: i64,
    /// Number of times this item has been re-extracted/updated.
    update_count: u32,
}

impl MemoryItem {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        kind: MemoryKind,
        name: String,
        content: String,
        source_session_id: Option<String>,
        project: Option<String>,
        created_at: i64,
        updated_at: i64,
        update_count: u32,
    ) -> Self {
        Self {
            id,
            kind,
            name,
            content,
            source_session_id,
            project,
            created_at,
            updated_at,
            update_count,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn kind(&self) -> MemoryKind {
        self.kind
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn source_session_id(&self) -> Option<&str> {
        self.source_session_id.as_deref()
    }

    /// Project, or `None` for a global memory.
    pub fn project(&self) -> Option<&str> {
        self.project.as_deref()
    }

    pub fn created_at(&self) -> i64 {
        self.created_at
    }

    pub fn updated_at(&self) -> i64 {
        self.updated_at
    }

    pub fn update_count(&self) -> u32 {
        self.update_count
    }
}

/// One message of an imported session transcript, normalized to the minimum
/// the extraction model needs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    /// `user`, `assistant`, or `system`.
    pub role: String,
    /// Text content. Tool activity is summarized inline as
    /// `ToolCall: name=...; input=...` lines by the transcript parser.
    pub content: String,
    /// ISO-8601 timestamp when available.
    pub timestamp: Option<String>,
}

/// A finished session transcript, ready for memory extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTranscript {
    /// Stable session identifier (used for idempotent imports).
    pub id: String,
    /// Where the transcript came from (file path or external ID).
    pub source: String,
    /// Project the session ran in — the repository/working-directory name (not
    /// the full path), when known. Passed to extraction so project-specific
    /// memories can be scoped to it. `None` when the source did not record a
    /// working directory.
    #[serde(default)]
    pub project: Option<String>,
    pub messages: Vec<SessionMessage>,
}

impl SessionTranscript {
    /// Timestamp of the first message that carries one.
    pub fn started_at(&self) -> Option<&str> {
        self.messages.iter().find_map(|m| m.timestamp.as_deref())
    }

    /// Timestamp of the last message that carries one.
    pub fn ended_at(&self) -> Option<&str> {
        self.messages
            .iter()
            .rev()
            .find_map(|m| m.timestamp.as_deref())
    }
}

/// Record of a session that has been imported into the memory store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportedSession {
    pub id: String,
    pub source: String,
    pub imported_at: i64,
    pub message_count: usize,
    /// Number of memory items written (created or updated) by the extraction.
    pub items_written: usize,
    /// Why this session was recorded. A failed attempt is recorded too, so the
    /// dream harvest stops retrying it every cycle; see [`SessionStatus`].
    pub status: SessionStatus,
    /// Error that made the attempt fail, for `SessionStatus::Failed`.
    pub last_error: Option<String>,
}

/// Outcome of an attempt to import a session.
///
/// Both outcomes act as "do not harvest this again" markers. Without a marker
/// for failures, a session that can never import — an unreadable transcript, or
/// one a small model always fails to produce valid JSON for — is retried by
/// every scheduled dream cycle forever, burning an LLM call each time.
/// Recording the failure ends that loop while leaving the session available for
/// a deliberate manual retry (`codesearch memory import <path> --force`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionStatus {
    Imported,
    Failed,
}

impl SessionStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            SessionStatus::Imported => "imported",
            SessionStatus::Failed => "failed",
        }
    }

    /// Unknown values read back from an older database default to `Imported`,
    /// matching how those rows were written before the column existed.
    pub fn parse(s: &str) -> SessionStatus {
        match s.trim().to_ascii_lowercase().as_str() {
            "failed" => SessionStatus::Failed,
            _ => SessionStatus::Imported,
        }
    }
}

/// Kind of a node in the memory virtual filesystem.
///
/// There are three top-level context types (`memory`, `session`, `resource`).
/// Nodes are the *navigable* layer over the flat [`MemoryItem`] store: each
/// node carries a short L0 abstract and a longer L1 overview so an agent can
/// read the summary first and drill into detail (`content`, the L2 layer) only
/// when needed.
///
/// - `Memory` — the whole-memory digest (`memory://memory`): a regenerated
///   abstract + overview over every stored [`MemoryItem`], read first before
///   drilling into individual memories.
/// - `Project` — the digest of one project/namespace
///   (`memory://projects/<project>`): a regenerated abstract + overview over
///   the items belonging to that project, read first when working in it.
/// - `Session` — one imported session (`memory://sessions/<id>`): its L2 is
///   the full normalized transcript, kept so the conversation can be re-read.
/// - `Resource` — a file or URL added explicitly via `memory add`
///   (`memory://resources/...`); its L2 is the fetched text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeKind {
    Memory,
    Project,
    Session,
    Resource,
}

impl NodeKind {
    pub const ALL: [NodeKind; 4] = [
        NodeKind::Memory,
        NodeKind::Project,
        NodeKind::Session,
        NodeKind::Resource,
    ];

    /// Stable identifier used in storage.
    pub fn as_str(&self) -> &'static str {
        match self {
            NodeKind::Memory => "memory",
            NodeKind::Project => "project",
            NodeKind::Session => "session",
            NodeKind::Resource => "resource",
        }
    }

    pub fn parse(s: &str) -> Option<NodeKind> {
        match s.trim().to_ascii_lowercase().as_str() {
            "memory" => Some(NodeKind::Memory),
            "project" => Some(NodeKind::Project),
            "session" => Some(NodeKind::Session),
            "resource" => Some(NodeKind::Resource),
            _ => None,
        }
    }
}

impl std::fmt::Display for NodeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A node in the memory virtual filesystem, addressed by a `memory://` URI.
///
/// Each node bundles three context levels for one location:
/// L0 `abstract` (the one-line summary retrieval ranks on), L1 `overview`
/// (a paragraph/outline to orient before reading), and L2 `content` (the full
/// detail — e.g. a session's transcript). `content` is empty for pure index
/// nodes such as the memory digest, whose value is entirely in L0/L1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    /// `memory://` URI uniquely identifying this node (also the primary key).
    uri: String,
    kind: NodeKind,
    /// URI of the parent directory, or `None` for a filesystem root.
    parent_uri: Option<String>,
    /// Human-readable display name, when the URI slug isn't presentable. For a
    /// project digest this is the original project string (e.g. the git remote
    /// `github.com/org/repo`), which the URI slugifies lossily. `None` for nodes
    /// whose URI's last component is already a fine label (sessions, resources).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    label: Option<String>,
    /// L0 — one-line summary; what recall returns and ranks on.
    abstract_: String,
    /// L1 — a paragraph or outline orienting the reader before L2.
    overview: String,
    /// L2 — full detail (e.g. a session transcript). Empty for index nodes.
    content: String,
    created_at: i64,
    updated_at: i64,
}

impl MemoryNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        uri: String,
        kind: NodeKind,
        parent_uri: Option<String>,
        abstract_: String,
        overview: String,
        content: String,
        created_at: i64,
        updated_at: i64,
    ) -> Self {
        Self {
            uri,
            kind,
            parent_uri,
            label: None,
            abstract_,
            overview,
            content,
            created_at,
            updated_at,
        }
    }

    /// Set the display label (builder-style), for nodes whose URI slug isn't a
    /// good human name (project digests carry their original project string).
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        let label = label.into();
        self.label = if label.is_empty() { None } else { Some(label) };
        self
    }

    pub fn uri(&self) -> &str {
        &self.uri
    }

    /// The display label if set, else `None` (callers fall back to the URI).
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub fn kind(&self) -> NodeKind {
        self.kind
    }

    pub fn parent_uri(&self) -> Option<&str> {
        self.parent_uri.as_deref()
    }

    pub fn abstract_(&self) -> &str {
        &self.abstract_
    }

    pub fn overview(&self) -> &str {
        &self.overview
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn created_at(&self) -> i64 {
        self.created_at
    }

    pub fn updated_at(&self) -> i64 {
        self.updated_at
    }

    /// Text used to build the node's L0 embedding — the abstract plus a short
    /// tail of the overview, so semantic recall matches on the summary.
    pub fn embedding_text(&self) -> String {
        if self.overview.trim().is_empty() {
            self.abstract_.clone()
        } else {
            format!("{}\n\n{}", self.abstract_, self.overview)
        }
    }
}

/// Record of one completed dream cycle — the pass that harvests finished
/// sessions and reorganizes the memory store.
///
/// Stored so the next cycle can tell whether anything changed since the last
/// one (and skip itself when nothing did), and so users can inspect what
/// dreaming has been doing (`memory dream --status`, `GET /api/memory/dream`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamRun {
    pub id: String,
    pub started_at: i64,
    pub finished_at: i64,
    /// Finished sessions discovered and imported by the harvest phase.
    pub sessions_imported: usize,
    /// Near-duplicate/contradiction clusters examined by consolidation.
    pub clusters_found: usize,
    /// Memory operations applied across all phases.
    pub operations_applied: usize,
    /// Operations proposed by the model but rejected by a guardrail.
    pub operations_skipped: usize,
    /// Outcome of the cycle: `"completed"`, or `"failed: <reason>"` when a
    /// phase errored after earlier phases may have already written. Recorded
    /// so a partially-applied cycle still leaves an inspectable trace.
    pub status: String,
}

/// A single write/delete decided by the extraction model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryOperation {
    /// Create or rewrite the item identified by `(kind, name)`.
    Upsert {
        kind: MemoryKind,
        name: String,
        content: String,
        /// Project this memory is specific to, or `None` if it applies
        /// globally. Set by the extraction model per item.
        project: Option<String>,
    },
    /// Remove the item identified by `(kind, name)`.
    Delete { kind: MemoryKind, name: String },
}
