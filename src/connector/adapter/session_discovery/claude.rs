//! Discovery of Claude Code sessions from `~/.claude/projects/**/*.jsonl`.

use std::path::Path;

use serde_json::Value;
use walkdir::WalkDir;

use crate::domain::{
    approx_tokens_from_chars, DiscoveredSession, DomainError, SessionLocator, SessionSource,
};

use super::{home_dir, parse_iso8601_secs, tail_preview, truncate_chars};

/// Maximum title length kept from a summary / first user message.
const MAX_TITLE_CHARS: usize = 80;

/// Number of trailing text messages scanned to build the tail preview.
const PREVIEW_MESSAGES: usize = 6;

/// List Claude Code sessions. Returns an empty list when the projects
/// directory does not exist (Claude Code not installed / never used).
pub fn discover() -> Result<Vec<DiscoveredSession>, DomainError> {
    let root = home_dir()?.join(".claude").join("projects");
    if !root.exists() {
        return Ok(Vec::new());
    }

    let mut sessions = Vec::new();
    for entry in WalkDir::new(&root)
        .max_depth(2)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "jsonl"))
    {
        match summarize_file(entry.path()) {
            Ok(Some(session)) => sessions.push(session),
            Ok(None) => {}
            Err(e) => tracing::debug!("skipping claude session {:?}: {e}", entry.path()),
        }
    }
    Ok(sessions)
}

/// Cheaply summarize one JSONL file into a [`DiscoveredSession`] without a full
/// parse: pull the session id, a title, message count, and a tail preview.
fn summarize_file(path: &Path) -> Result<Option<DiscoveredSession>, DomainError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| DomainError::invalid_input(format!("cannot read {}: {e}", path.display())))?;

    let mut session_id: Option<String> = None;
    let mut summary: Option<String> = None;
    let mut first_user: Option<String> = None;
    let mut last_texts: Vec<String> = Vec::new();
    let mut message_count = 0usize;
    let mut total_text_chars = 0usize;
    let mut last_timestamp: Option<String> = None;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };

        if session_id.is_none() {
            if let Some(id) = value.get("sessionId").and_then(Value::as_str) {
                session_id = Some(id.to_string());
            }
        }
        // Claude writes a `{"type":"summary","summary":"..."}` line for named
        // sessions — the nicest available title.
        if value.get("type").and_then(Value::as_str) == Some("summary") {
            if let Some(s) = value.get("summary").and_then(Value::as_str) {
                summary = Some(s.to_string());
            }
            continue;
        }

        let kind = value.get("type").and_then(Value::as_str);
        if kind != Some("user") && kind != Some("assistant") {
            continue;
        }
        if value.get("isMeta").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        let Some(message) = value.get("message") else {
            continue;
        };
        let Some(text) = message.get("content").and_then(render_text) else {
            continue;
        };
        if looks_like_machine_text(&text) {
            continue;
        }

        message_count += 1;
        total_text_chars += text.chars().count();
        if let Some(ts) = value.get("timestamp").and_then(Value::as_str) {
            last_timestamp = Some(ts.to_string());
        }
        if kind == Some("user") && first_user.is_none() {
            first_user = Some(text.clone());
        }
        push_bounded(&mut last_texts, text, PREVIEW_MESSAGES);
    }

    if message_count == 0 {
        return Ok(None);
    }

    let id = session_id.unwrap_or_else(|| {
        path.file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    });

    let title = summary
        .or_else(|| first_user.clone())
        .map(|t| {
            truncate_chars(
                &t.split_whitespace().collect::<Vec<_>>().join(" "),
                MAX_TITLE_CHARS,
            )
        })
        .unwrap_or_default();

    let updated_at = last_timestamp
        .as_deref()
        .and_then(parse_iso8601_secs)
        .unwrap_or_else(|| file_mtime_secs(path));

    let cwd = decode_project_dir(path);

    Ok(Some(DiscoveredSession {
        source: SessionSource::Claude,
        id,
        title,
        cwd,
        updated_at,
        message_count,
        approx_tokens: approx_tokens_from_chars(SessionSource::Claude, total_text_chars),
        tail_preview: tail_preview(&last_texts),
        locator: SessionLocator::File(path.to_string_lossy().to_string()),
    }))
}

/// Render a message `content` value (string or block array) to plain text,
/// keeping only text blocks (tool activity is elided for the preview).
fn render_text(content: &Value) -> Option<String> {
    let text = match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| {
                if b.get("type").and_then(Value::as_str) == Some("text") {
                    b.get("text").and_then(Value::as_str).map(str::to_string)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => return None,
    };
    let text = text.trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

fn push_bounded(buf: &mut Vec<String>, text: String, max: usize) {
    buf.push(text);
    if buf.len() > max {
        buf.remove(0);
    }
}

/// Decode a Claude project directory name back into a filesystem path.
/// Claude encodes the cwd by replacing `/` with `-`, so `-Users-me-proj`
/// becomes `/Users/me/proj` (best-effort — lossy for names containing `-`).
fn decode_project_dir(path: &Path) -> Option<String> {
    let dir = path.parent()?.file_name()?.to_str()?;
    if dir.starts_with('-') {
        Some(dir.replace('-', "/"))
    } else {
        Some(dir.to_string())
    }
}

fn looks_like_machine_text(text: &str) -> bool {
    let head = text.trim_start();
    head.starts_with("<command-name>")
        || head.starts_with("<command-message>")
        || head.starts_with("<local-command-stdout>")
        || head.starts_with("<system-reminder>")
        || head.starts_with("[Request interrupted")
        || head.starts_with("Caveat: The messages below were generated")
}

fn file_mtime_secs(path: &Path) -> i64 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_project_dir() {
        let p = Path::new("/home/u/.claude/projects/-Users-me-proj/s.jsonl");
        assert_eq!(decode_project_dir(p).as_deref(), Some("/Users/me/proj"));
    }
}
