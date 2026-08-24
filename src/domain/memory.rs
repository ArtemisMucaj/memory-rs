use serde::{Deserialize, Serialize};

/// Category of a stored memory.
///
/// Single-variant on purpose: the previous taxonomy (Preference / Experience /
/// Skill / Fact) was unreliable to extract and added prompt surface without
/// paying for itself — `Skill` and `Experience` were the two the extraction
/// model most often hallucinated structure onto. The enum stays so the
/// storage column remains forward-compatible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryKind {
    Fact,
}

impl MemoryKind {
    pub const ALL: [MemoryKind; 1] = [MemoryKind::Fact];

    /// Stable identifier used in storage and in the extraction JSON protocol.
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryKind::Fact => "fact",
        }
    }

    /// Plural field name used in extraction output JSON.
    pub fn plural(&self) -> &'static str {
        match self {
            MemoryKind::Fact => "facts",
        }
    }

    /// Title-cased plural, used for group headers in listings.
    pub fn plural_title(&self) -> &'static str {
        match self {
            MemoryKind::Fact => "Facts",
        }
    }

    pub fn parse(s: &str) -> Option<MemoryKind> {
        match s.trim().to_ascii_lowercase().as_str() {
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportedSession {
    pub id: String,
    pub source: String,
    pub imported_at: i64,
    pub message_count: usize,
    /// The project the session was worked in, when the transcript names one.
    ///
    /// Recorded on the session rather than derived from the memories it wrote,
    /// because a session that produced nothing durable is still part of the
    /// answer to "what was I working on" — and deriving it would make exactly
    /// those sessions invisible.
    pub project: Option<String>,
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
/// a deliberate manual retry (`memory-rs import <path> --force`).
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
