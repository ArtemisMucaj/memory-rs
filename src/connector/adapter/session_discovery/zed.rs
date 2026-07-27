//! Discovery of Zed assistant threads from
//! `~/Library/Application Support/Zed/threads/threads.db`.
//!
//! Each `threads` row has a `summary` (nice name) and a `data` BLOB that is
//! **zstd-compressed** JSON of the shape:
//! `{"title": …, "messages": [{"User"|"Agent": {"content": [{"Text": "…"} | {"Thinking": {…}} | …]}}]}`.
//! We decompress on demand and keep the `Text` blocks as the transcript.

use rusqlite::{Connection, OpenFlags};
use serde_json::Value;

use crate::domain::{
    approx_tokens_from_chars, DiscoveredSession, DomainError, SessionLocator, SessionMessage,
    SessionSource, SessionTranscript,
};

use super::{home_dir, parse_iso8601_secs, tail_preview, truncate_chars};

const MAX_TITLE_CHARS: usize = 80;
const PREVIEW_MESSAGES: usize = 6;

fn db_path() -> Result<std::path::PathBuf, DomainError> {
    Ok(home_dir()?.join("Library/Application Support/Zed/threads/threads.db"))
}

pub fn discover() -> Result<Vec<DiscoveredSession>, DomainError> {
    let path = db_path()?;
    if !path.exists() {
        return Ok(Vec::new());
    }
    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| DomainError::storage(format!("cannot open Zed threads.db: {e}")))?;
    let db_path_str = path.to_string_lossy().to_string();

    let mut stmt = conn
        .prepare(
            "SELECT id, summary, updated_at, data_type, data \
             FROM threads ORDER BY updated_at DESC",
        )
        .map_err(|e| DomainError::storage(format!("Zed query prepare failed: {e}")))?;

    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1).unwrap_or_default(),
                row.get::<_, String>(2).unwrap_or_default(),
                row.get::<_, String>(3).unwrap_or_default(),
                row.get::<_, Vec<u8>>(4)?,
            ))
        })
        .map_err(|e| DomainError::storage(format!("Zed query failed: {e}")))?;

    let mut sessions = Vec::new();
    for row in rows {
        let (id, summary, updated_at, data_type, blob) =
            row.map_err(|e| DomainError::storage(format!("Zed row read failed: {e}")))?;

        let texts = match thread_texts(&blob, &data_type) {
            Ok(t) => t,
            Err(e) => {
                tracing::debug!("skipping Zed thread {id}: {e}");
                continue;
            }
        };
        if texts.is_empty() {
            continue;
        }
        let total_chars: usize = texts.iter().map(|t| t.chars().count()).sum();
        let tail: Vec<String> = texts
            .iter()
            .rev()
            .take(PREVIEW_MESSAGES)
            .rev()
            .cloned()
            .collect();

        sessions.push(DiscoveredSession {
            source: SessionSource::Zed,
            title: truncate_chars(summary.trim(), MAX_TITLE_CHARS),
            cwd: None,
            updated_at: parse_iso8601_secs(&updated_at).unwrap_or(0),
            message_count: texts.len(),
            approx_tokens: approx_tokens_from_chars(SessionSource::Zed, total_chars),
            tail_preview: tail_preview(&tail),
            locator: SessionLocator::Sqlite {
                db_path: db_path_str.clone(),
                session_id: id.clone(),
            },
            id,
        });
    }
    Ok(sessions)
}

pub fn load_transcript(db_path: &str, session_id: &str) -> Result<SessionTranscript, DomainError> {
    let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| DomainError::storage(format!("cannot open Zed threads.db: {e}")))?;

    let (data_type, blob): (String, Vec<u8>) = conn
        .query_row(
            "SELECT data_type, data FROM threads WHERE id = ?1",
            [session_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|e| DomainError::storage(format!("Zed thread '{session_id}' not found: {e}")))?;

    let messages = thread_messages(&blob, &data_type)?;
    if messages.is_empty() {
        return Err(DomainError::invalid_input(format!(
            "Zed thread '{session_id}' has no messages"
        )));
    }
    Ok(SessionTranscript {
        id: session_id.to_string(),
        source: format!("zed:{session_id}"),
        // Scope is set by the discovery dispatcher from the session's cwd.
        project: None,
        messages,
    })
}

/// Decompress a thread blob (zstd when `data_type` says so; otherwise assume
/// raw JSON) and parse it into a serde `Value`.
fn decode_thread(blob: &[u8], data_type: &str) -> Result<Value, DomainError> {
    let json_bytes = if data_type.eq_ignore_ascii_case("zstd") {
        zstd::stream::decode_all(blob)
            .map_err(|e| DomainError::parse(format!("Zed zstd decode failed: {e}")))?
    } else {
        blob.to_vec()
    };
    serde_json::from_slice(&json_bytes)
        .map_err(|e| DomainError::parse(format!("Zed thread JSON parse failed: {e}")))
}

/// Extract the ordered `SessionMessage`s from a thread blob.
fn thread_messages(blob: &[u8], data_type: &str) -> Result<Vec<SessionMessage>, DomainError> {
    let value = decode_thread(blob, data_type)?;
    let messages = value
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| DomainError::parse("Zed thread has no messages array"))?;

    let mut out = Vec::new();
    for msg in messages {
        // Each message is a single-key tagged object: {"User": {...}} / {"Agent": {...}}.
        let Some((tag, body)) = msg.as_object().and_then(|o| o.iter().next()) else {
            continue;
        };
        let role = match tag.as_str() {
            "User" => "user",
            "Agent" => "assistant",
            _ => continue,
        };
        let text = content_text(body.get("content"));
        if text.trim().is_empty() {
            continue;
        }
        out.push(SessionMessage {
            role: role.to_string(),
            content: text,
            timestamp: None,
        });
    }
    Ok(out)
}

/// Plain text of every message (for previews / counting).
fn thread_texts(blob: &[u8], data_type: &str) -> Result<Vec<String>, DomainError> {
    Ok(thread_messages(blob, data_type)?
        .into_iter()
        .map(|m| m.content)
        .collect())
}

/// Render a message's `content` array (`[{"Text": "…"} | {"Thinking": {…}} | …]`)
/// to plain text, keeping user/assistant prose and eliding tool/thinking noise.
fn content_text(content: Option<&Value>) -> String {
    let Some(blocks) = content.and_then(Value::as_array) else {
        return String::new();
    };
    let mut parts = Vec::new();
    for block in blocks {
        // Blocks are tagged: {"Text": "…"} or {"Thinking": {"text": "…"}}, etc.
        if let Some(s) = block.get("Text").and_then(Value::as_str) {
            if !s.trim().is_empty() {
                parts.push(s.trim().to_string());
            }
        } else if let Some(obj) = block.as_object() {
            // Tool-use blocks: keep a one-line marker as evidence.
            if let Some(tag) = obj.keys().next() {
                if tag == "ToolUse" || tag == "ToolResult" {
                    parts.push(format!("ToolCall: {tag}"));
                }
            }
        }
    }
    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tagged_messages() {
        let json = serde_json::json!({
            "messages": [
                {"User": {"content": [{"Text": "hello there"}]}},
                {"Agent": {"content": [
                    {"Thinking": {"text": "hmm"}},
                    {"Text": "hi back"}
                ]}}
            ]
        });
        let bytes = serde_json::to_vec(&json).unwrap();
        let msgs = thread_messages(&bytes, "json").unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].content, "hello there");
        assert_eq!(msgs[1].role, "assistant");
        // Thinking is elided; only Text is kept.
        assert_eq!(msgs[1].content, "hi back");
    }

    #[test]
    fn zstd_roundtrip_decodes() {
        let json = serde_json::json!({
            "messages": [{"User": {"content": [{"Text": "compressed hi"}]}}]
        });
        let raw = serde_json::to_vec(&json).unwrap();
        let compressed = zstd::stream::encode_all(&raw[..], 0).unwrap();
        let msgs = thread_messages(&compressed, "zstd").unwrap();
        assert_eq!(msgs[0].content, "compressed hi");
    }
}
