//! Discovery of OpenCode sessions from `~/.local/share/opencode/opencode.db`.
//!
//! OpenCode stores each session's metadata in a `session` row (with a `title`)
//! and its conversation as `message` rows (role) whose text lives in `part`
//! rows (`{"type":"text","text":…}`). A transcript is the ordered join of the
//! two by `time_created`.

use rusqlite::{Connection, OpenFlags};
use serde_json::Value;

use crate::domain::{
    approx_tokens_from_chars, DiscoveredSession, DomainError, SessionLocator, SessionMessage,
    SessionSource, SessionTranscript,
};

use super::{home_dir, truncate_chars};

const MAX_TITLE_CHARS: usize = 80;

/// Open the OpenCode database read-only, or `Ok(None)` when it is absent.
fn open_db() -> Result<Option<Connection>, DomainError> {
    let path = home_dir()?.join(".local/share/opencode/opencode.db");
    if !path.exists() {
        return Ok(None);
    }
    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| DomainError::storage(format!("cannot open opencode.db: {e}")))?;
    Ok(Some(conn))
}

pub fn discover() -> Result<Vec<DiscoveredSession>, DomainError> {
    let Some(conn) = open_db()? else {
        return Ok(Vec::new());
    };
    let db_path = home_dir()?
        .join(".local/share/opencode/opencode.db")
        .to_string_lossy()
        .to_string();

    // Discovery lists sessions CHEAPLY, like the Zed source: a single query over
    // the `session` table, with no message/part reads. `message_count` comes from
    // an index-backed COUNT subquery; the message body (needed only for the
    // right-pane transcript) is materialized lazily by `load_transcript` when a
    // session is actually highlighted. This avoids the previous per-session
    // message/part queries that made OpenCode discovery slow for large histories.
    let mut stmt = conn
        .prepare(
            "SELECT s.id, s.title, s.directory, s.time_updated, \
                    (SELECT COUNT(*) FROM message m WHERE m.session_id = s.id) AS msg_count, \
                    (SELECT COALESCE(SUM(LENGTH(json_extract(p.data, '$.text'))), 0) \
                       FROM message m JOIN part p ON p.message_id = m.id \
                       WHERE m.session_id = s.id) AS total_chars \
              FROM session s \
              WHERE s.time_archived IS NULL \
                AND EXISTS (SELECT 1 FROM message m WHERE m.session_id = s.id) \
              ORDER BY s.time_updated DESC",
        )
        .map_err(|e| DomainError::storage(format!("opencode query prepare failed: {e}")))?;

    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1).unwrap_or_default(),
                row.get::<_, Option<String>>(2)?,
                row.get::<_, i64>(3).unwrap_or(0),
                row.get::<_, i64>(4).unwrap_or(0),
                row.get::<_, i64>(5).unwrap_or(0),
            ))
        })
        .map_err(|e| DomainError::storage(format!("opencode query failed: {e}")))?;

    let mut sessions = Vec::new();
    for row in rows {
        let (id, title, directory, time_updated_ms, msg_count, total_chars) =
            row.map_err(|e| DomainError::storage(format!("opencode row read failed: {e}")))?;

        sessions.push(DiscoveredSession {
            source: SessionSource::OpenCode,
            title: truncate_chars(title.trim(), MAX_TITLE_CHARS),
            cwd: directory,
            updated_at: time_updated_ms / 1000,
            message_count: msg_count.max(0) as usize,
            approx_tokens: approx_tokens_from_chars(
                SessionSource::OpenCode,
                total_chars.max(0) as usize,
            ),
            // Preview is not shown in the picker; leave it empty and let the
            // full transcript load lazily on selection.
            tail_preview: String::new(),
            locator: SessionLocator::Sqlite {
                db_path: db_path.clone(),
                session_id: id.clone(),
            },
            id,
        });
    }

    Ok(sessions)
}

/// Load the full transcript for one OpenCode session.
pub fn load_transcript(db_path: &str, session_id: &str) -> Result<SessionTranscript, DomainError> {
    let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| DomainError::storage(format!("cannot open opencode.db: {e}")))?;

    let messages = ordered_messages(&conn, session_id)?;
    if messages.is_empty() {
        return Err(DomainError::invalid_input(format!(
            "opencode session '{session_id}' has no messages"
        )));
    }

    Ok(SessionTranscript {
        id: session_id.to_string(),
        source: format!("opencode:{session_id}"),
        // Scope is set by the discovery dispatcher from the session's cwd.
        project: None,
        messages,
    })
}

/// Build the ordered `SessionMessage` list in a SINGLE query: join `message`
/// to its `part`s, ordered by (message time, part time), and group rows into
/// messages in one pass. This avoids the N+1 pattern of one `part` query per
/// message — important for OpenCode histories with many messages.
fn ordered_messages(
    conn: &Connection,
    session_id: &str,
) -> Result<Vec<SessionMessage>, DomainError> {
    let mut stmt = conn
        .prepare(
            "SELECT m.id, m.data, p.data \
             FROM message m \
             LEFT JOIN part p ON p.message_id = m.id \
             WHERE m.session_id = ?1 \
             ORDER BY m.time_created ASC, p.time_created ASC",
        )
        .map_err(|e| DomainError::storage(format!("opencode message prepare failed: {e}")))?;
    let rows = stmt
        .query_map([session_id], |row| {
            Ok((
                row.get::<_, String>(0)?,         // message id
                row.get::<_, String>(1)?,         // message data (role)
                row.get::<_, Option<String>>(2)?, // part data (NULL if none)
            ))
        })
        .map_err(|e| DomainError::storage(format!("opencode message query failed: {e}")))?;

    // Group consecutive rows by message id (the ORDER BY keeps them contiguous).
    let mut messages = Vec::new();
    let mut current_id: Option<String> = None;
    let mut role = String::new();
    let mut parts: Vec<String> = Vec::new();

    let flush = |messages: &mut Vec<SessionMessage>, role: &str, parts: &[String]| {
        let content = parts.join("\n");
        if !content.trim().is_empty() {
            messages.push(SessionMessage {
                role: role.to_string(),
                content,
                timestamp: None,
            });
        }
    };

    for row in rows {
        let (msg_id, msg_data, part_data) =
            row.map_err(|e| DomainError::storage(format!("opencode row read failed: {e}")))?;
        if current_id.as_deref() != Some(&msg_id) {
            if current_id.is_some() {
                flush(&mut messages, &role, &parts);
            }
            current_id = Some(msg_id);
            role = serde_json::from_str::<Value>(&msg_data)
                .ok()
                .and_then(|v| v.get("role").and_then(Value::as_str).map(str::to_string))
                .unwrap_or_else(|| "assistant".to_string());
            parts.clear();
        }
        if let Some(data) = part_data {
            if let Some(rendered) = render_part(&data) {
                parts.push(rendered);
            }
        }
    }
    if current_id.is_some() {
        flush(&mut messages, &role, &parts);
    }
    Ok(messages)
}

/// Render one `part` row's JSON into transcript text, or `None` to skip it.
/// `text` parts contribute their trimmed text; `tool` parts become a compact
/// `ToolCall:` line matching the Claude parser's format.
fn render_part(data: &str) -> Option<String> {
    let value = serde_json::from_str::<Value>(data).ok()?;
    match value.get("type").and_then(Value::as_str) {
        Some("text") => value
            .get("text")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(str::to_string),
        Some("tool") => {
            let name = value
                .get("tool")
                .and_then(Value::as_str)
                .or_else(|| value.get("name").and_then(Value::as_str))
                .unwrap_or("tool");
            // The tool's arguments live in `state.input`; render them compactly
            // so the transcript shows *what* the tool was asked to do.
            let input = value
                .get("state")
                .and_then(|s| s.get("input"))
                .map(render_tool_input)
                .filter(|s| !s.is_empty());
            Some(match input {
                Some(args) => format!("ToolCall: name={name}; input={args}"),
                None => format!("ToolCall: name={name}"),
            })
        }
        _ => None,
    }
}

/// Maximum characters of a rendered tool input (matches the Claude parser).
const MAX_TOOL_INPUT_CHARS: usize = 200;

/// Compact single-line rendering of a tool input object (`k=v, k=v`), truncated.
fn render_tool_input(input: &Value) -> String {
    let rendered = match input {
        Value::Object(map) => map
            .iter()
            .map(|(k, v)| {
                let v = match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                format!("{k}={v}")
            })
            .collect::<Vec<_>>()
            .join(", "),
        other => other.to_string(),
    };
    let compact: String = rendered.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() > MAX_TOOL_INPUT_CHARS {
        let kept: String = compact.chars().take(MAX_TOOL_INPUT_CHARS).collect();
        format!("{kept}…")
    } else {
        compact
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_tool_input_args() {
        let input = serde_json::json!({"command": "git status", "timeout": 30000});
        let rendered = render_tool_input(&input);
        assert!(rendered.contains("command=git status"));
        assert!(rendered.contains("timeout=30000"));
    }

    #[test]
    fn truncates_long_tool_input() {
        let input = serde_json::json!({"command": "x".repeat(500)});
        assert!(render_tool_input(&input).chars().count() <= MAX_TOOL_INPUT_CHARS + 1);
    }
}
