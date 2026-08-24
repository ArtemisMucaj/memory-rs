//! Discovery of finished assistant sessions from local tools, for the
//! `memory import` picker.
//!
//! Each source knows how to (1) list its sessions cheaply — friendly name,
//! timestamps, and a short preview from the *end* of the conversation — and
//! (2) materialize a full [`SessionTranscript`](crate::domain::SessionTranscript)
//! on demand, only for the sessions the user actually selects to import.
//!
//! Sources:
//! - [`claude`] — `~/.claude/projects/**/*.jsonl` (reuses the JSONL parser).
//! - [`opencode`] — `~/.local/share/opencode/opencode.db` (SQLite).
//! - [`zed`] — `~/Library/Application Support/Zed/threads/threads.db`
//!   (SQLite; thread bodies are zstd-compressed).

mod claude;
mod opencode;
mod zed;

use crate::connector::adapter::project::project_from_cwd;
use crate::connector::adapter::transcript::parse_transcript_file;
use crate::domain::{
    DiscoveredSession, DomainError, SessionLocator, SessionSource, SessionTranscript,
};

/// Sentinel for session discovery sources.
pub type DiscoveredSessionSources = Vec<DiscoveredSession>;

/// [`SessionDiscovery`](crate::application::SessionDiscovery) adapter over the local session
/// stores, used by the dream use case to harvest finished sessions. Discovery
/// and transcript loading are blocking file/SQLite I/O, so both are pushed off
/// the async runtime via `spawn_blocking`.
pub struct LocalSessionDiscovery {
    /// Metadata database consulted to map a session's working directory to
    /// the namespace it was indexed under, so its memories carry that
    /// namespace as their project. `None` skips the lookup (memories fall back
    /// to the git remote, otherwise global).
    db_path: Option<std::path::PathBuf>,
}

impl LocalSessionDiscovery {
    pub fn new(db_path: Option<std::path::PathBuf>) -> Self {
        Self { db_path }
    }
}

#[async_trait::async_trait]
impl crate::application::SessionDiscovery for LocalSessionDiscovery {
    async fn discover(&self) -> Result<Vec<DiscoveredSession>, DomainError> {
        tokio::task::spawn_blocking(discover_all_sessions)
            .await
            .map_err(|e| DomainError::internal(format!("session discovery task panicked: {e}")))
    }

    async fn load_transcript(
        &self,
        session: &DiscoveredSession,
    ) -> Result<SessionTranscript, DomainError> {
        let owned = session.clone();
        let db_path = self.db_path.clone();
        tokio::task::spawn_blocking(move || load_transcript(&owned, db_path.as_deref()))
            .await
            .map_err(|e| DomainError::internal(format!("transcript load task panicked: {e}")))?
    }

    fn session_project(&self, session: &DiscoveredSession) -> Option<String> {
        // Same derivation `load_transcript` applies, so a session's harvest
        // eligibility and the project its memories land under always agree.
        session.cwd.as_deref().and_then(project_from_cwd)
    }
}

/// Characters of end-of-session preview surfaced in the picker.
pub(crate) const PREVIEW_CHARS: usize = 240;

/// The discovery sources, each as a `(name, discover_fn)` pair. Iterating this
/// keeps the streaming and blocking entry points in sync.
type SourceFn = fn() -> Result<Vec<DiscoveredSession>, DomainError>;
const SOURCES: [(&str, SourceFn); 3] = [
    ("claude", claude::discover),
    ("opencode", opencode::discover),
    ("zed", zed::discover),
];

/// Discover sessions from every available source, newest first.
///
/// A source that is not installed (its store is absent) simply contributes
/// nothing; a source that errors is logged and skipped, so one broken store
/// never blocks the picker.
pub fn discover_all_sessions() -> Vec<DiscoveredSession> {
    let mut sessions = Vec::new();
    for (name, discover) in SOURCES {
        match discover() {
            Ok(mut found) => sessions.append(&mut found),
            Err(e) => tracing::warn!("session discovery for {name} failed: {e}"),
        }
    }
    sessions.sort_by_key(|s| std::cmp::Reverse(s.updated_at));
    sessions
}

/// Materialize the full transcript for a discovered session, so it can be run
/// through the import pipeline. The session's working directory (when known)
/// is carried into the transcript as its memory project, resolved through the
/// session's cwd field (set by the discovery layer) and falling back to
/// whatever the transcript itself recorded.
pub fn load_transcript(
    session: &DiscoveredSession,
    _db_path: Option<&std::path::Path>,
) -> Result<SessionTranscript, DomainError> {
    let mut transcript = match (&session.source, &session.locator) {
        (SessionSource::Claude, SessionLocator::File(path)) => {
            parse_transcript_file(std::path::Path::new(path))
        }
        (
            SessionSource::OpenCode,
            SessionLocator::Sqlite {
                db_path,
                session_id,
            },
        ) => opencode::load_transcript(db_path, session_id),
        (
            SessionSource::Zed,
            SessionLocator::Sqlite {
                db_path,
                session_id,
            },
        ) => zed::load_transcript(db_path, session_id),
        (source, locator) => Err(DomainError::invalid_input(format!(
            "mismatched session source {source} and locator {locator:?}"
        ))),
    }?;

    // Normalize the session's working directory into a stable project name:
    // the git remote's `owner/repo` when the cwd is inside a repo, else the
    // directory basename, else global. A clean name (not a machine-specific
    // absolute path) is what the user assigns to a namespace.
    //
    // Only when discovery actually derived one: the parser may already have
    // scoped the transcript from the cwd recorded inside it, and overwriting
    // that with `None` would drop a session out of every scoped briefing.
    if let Some(project) = session.cwd.as_deref().and_then(project_from_cwd) {
        transcript.project = Some(project);
    }
    Ok(transcript)
}

/// The absolute path to `$HOME`, or an error when it cannot be determined.
pub(crate) fn home_dir() -> Result<std::path::PathBuf, DomainError> {
    std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .ok_or_else(|| DomainError::invalid_input("HOME environment variable is not set"))
}

/// Build a one-line, whitespace-collapsed preview from the tail of a message
/// list, truncated to [`PREVIEW_CHARS`].
pub(crate) fn tail_preview(messages: &[String]) -> String {
    // Walk from the newest message backward, appending prose to the preview so
    // it reflects how the session ended (newest first). Tool-call chatter is
    // stripped so a session that ended on a `bash` call still previews its
    // last human-readable outcome.
    let mut acc = String::new();
    for text in messages.iter().rev() {
        let line = strip_tool_markers(text);
        if line.is_empty() {
            continue;
        }
        if !acc.is_empty() {
            acc.push_str(" … ");
        }
        acc.push_str(&line);
        if acc.chars().count() >= PREVIEW_CHARS {
            break;
        }
    }
    truncate_chars(&acc, PREVIEW_CHARS)
}

/// Collapse whitespace and drop `ToolCall:` marker lines, keeping only prose.
fn strip_tool_markers(text: &str) -> String {
    text.lines()
        .filter(|l| !l.trim_start().starts_with("ToolCall:"))
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let kept: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    format!("{kept}…")
}

/// Parse the seconds component of an ISO-8601 timestamp without a date crate:
/// `YYYY-MM-DDThh:mm:ss…` → Unix seconds (UTC; sub-second and offset precision
/// ignored — good enough for sorting and "N ago").
pub(crate) fn parse_iso8601_secs(ts: &str) -> Option<i64> {
    if ts.len() < 19 {
        return None;
    }
    let num = |a: usize, b: usize| ts.get(a..b)?.parse::<i64>().ok();
    let year = num(0, 4)?;
    let month = num(5, 7)?;
    let day = num(8, 10)?;
    let hour = num(11, 13)?;
    let min = num(14, 16)?;
    let sec = num(17, 19)?;
    Some(days_from_civil(year, month, day) * 86400 + hour * 3600 + min * 60 + sec)
}

/// Days since the Unix epoch for a civil date (Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_preview_uses_end_of_session() {
        let msgs = vec![
            "first message".to_string(),
            "middle".to_string(),
            "the final outcome".to_string(),
        ];
        let p = tail_preview(&msgs);
        // The most recent message leads the preview.
        assert!(p.starts_with("the final outcome"));
    }

    #[test]
    fn tail_preview_truncates() {
        let long = "word ".repeat(200);
        let p = tail_preview(&[long]);
        assert!(p.chars().count() <= PREVIEW_CHARS);
    }

    #[test]
    fn parses_iso8601_to_epoch() {
        assert_eq!(parse_iso8601_secs("2026-07-01T10:00:00Z"), Some(1782900000));
        assert_eq!(parse_iso8601_secs("short"), None);
    }
}
