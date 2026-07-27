//! Parser for finished session transcripts.
//!
//! Supports two JSONL formats:
//!
//! 1. **Claude Code session logs** (`~/.claude/projects/<project>/<session>.jsonl`):
//!    each line is an event with a `type` (`user`, `assistant`, `summary`, …)
//!    and a nested `message` whose `content` is either a string or an array
//!    of blocks (`text`, `tool_use`, `tool_result`).
//! 2. **Generic chat logs**: each line is `{"role": "...", "content": "..."}`.
//!
//! The output is a normalized [`SessionTranscript`]: user/assistant text plus
//! one-line `ToolCall:` summaries of tool activity (evidence for experience
//! and skill extraction), with tool results omitted as too noisy.

use serde_json::Value;

use crate::domain::{DomainError, SessionMessage, SessionTranscript};

/// Maximum characters of a tool input rendered into a `ToolCall:` summary.
const MAX_TOOL_INPUT_CHARS: usize = 200;

/// Parse a transcript file (JSONL) into a [`SessionTranscript`].
pub fn parse_transcript_file(path: &std::path::Path) -> Result<SessionTranscript, DomainError> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        DomainError::invalid_input(format!("cannot read transcript '{}': {e}", path.display()))
    })?;
    let fallback_id = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown-session".to_string());
    parse_transcript(&content, &fallback_id, &path.display().to_string())
}

/// Parse JSONL transcript content.
///
/// `fallback_id` is used when no line carries a `sessionId`.
pub fn parse_transcript(
    content: &str,
    fallback_id: &str,
    source: &str,
) -> Result<SessionTranscript, DomainError> {
    let mut session_id: Option<String> = None;
    let mut cwd: Option<String> = None;
    let mut messages = Vec::new();
    let mut parsed_lines = 0usize;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        parsed_lines += 1;

        if session_id.is_none() {
            if let Some(id) = value.get("sessionId").and_then(Value::as_str) {
                session_id = Some(id.to_string());
            }
        }
        // Claude Code records the working directory per event; keep the first
        // one so `memory import <path>` (which bypasses discovery) is still
        // scoped to the project the session ran in.
        if cwd.is_none() {
            if let Some(dir) = value.get("cwd").and_then(Value::as_str) {
                cwd = Some(dir.to_string());
            }
        }

        if let Some(message) = parse_line(&value) {
            messages.push(message);
        }
    }

    if parsed_lines == 0 {
        return Err(DomainError::invalid_input(format!(
            "'{source}' contains no parseable JSONL lines"
        )));
    }

    Ok(SessionTranscript {
        id: session_id.unwrap_or_else(|| fallback_id.to_string()),
        source: source.to_string(),
        // The project is set by the session discovery layer from the session's
        // cwd. This parser leaves it `None` — the caller (discovery dispatcher)
        // resolves the cwd to a project name.
        project: None,
        messages,
    })
}

/// Parse one JSONL line into a normalized message, or `None` when the line
/// carries no conversational content (summaries, meta lines, snapshots, …).
fn parse_line(value: &Value) -> Option<SessionMessage> {
    // Claude Code format: { "type": "user"|"assistant", "message": {...} }.
    if let Some(kind) = value.get("type").and_then(Value::as_str) {
        if kind != "user" && kind != "assistant" {
            return None;
        }
        // Meta lines (command wrappers, hook output) are not user speech.
        if value.get("isMeta").and_then(Value::as_bool) == Some(true) {
            return None;
        }
        let message = value.get("message")?;
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or(kind)
            .to_string();
        let text = render_content(message.get("content")?)?;
        if looks_like_machine_text(&text) {
            return None;
        }
        return Some(SessionMessage {
            role,
            content: text,
            timestamp: value
                .get("timestamp")
                .and_then(Value::as_str)
                .map(String::from),
        });
    }

    // Generic format: { "role": "...", "content": "..." }.
    let role = value.get("role").and_then(Value::as_str)?.to_string();
    let text = render_content(value.get("content")?)?;
    Some(SessionMessage {
        role,
        content: text,
        timestamp: value
            .get("timestamp")
            .and_then(Value::as_str)
            .map(String::from),
    })
}

/// Render a message `content` value (string or block array) to plain text.
///
/// Text blocks are kept verbatim; `tool_use` blocks become one-line
/// `ToolCall:` summaries; `tool_result` and `thinking` blocks are dropped.
fn render_content(content: &Value) -> Option<String> {
    let text = match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => {
            let mut parts = Vec::new();
            for block in blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(t) = block.get("text").and_then(Value::as_str) {
                            parts.push(t.to_string());
                        }
                    }
                    Some("tool_use") => {
                        let name = block
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("unknown");
                        let input = block
                            .get("input")
                            .map(render_tool_input)
                            .unwrap_or_default();
                        parts.push(format!("ToolCall: name={name}; input={input}"));
                    }
                    _ => {}
                }
            }
            parts.join("\n")
        }
        _ => return None,
    };
    let text = text.trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

/// Compact single-line rendering of a tool input, truncated.
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
        let truncated: String = compact.chars().take(MAX_TOOL_INPUT_CHARS).collect();
        format!("{truncated}...")
    } else {
        compact
    }
}

/// Filter machine-generated wrapper text that Claude Code stores as user
/// messages (slash-command envelopes, interruption markers, hook output).
fn looks_like_machine_text(text: &str) -> bool {
    let head = text.trim_start();
    head.starts_with("<command-name>")
        || head.starts_with("<command-message>")
        || head.starts_with("<local-command-stdout>")
        || head.starts_with("<system-reminder>")
        || head.starts_with("[Request interrupted")
        || head.starts_with("Caveat: The messages below were generated")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_claude_code_format() {
        let content = r#"{"type":"summary","summary":"Session about testing"}
{"type":"user","sessionId":"abc-123","timestamp":"2026-07-01T10:00:00Z","message":{"role":"user","content":"Please fix the flaky test"}}
{"type":"assistant","sessionId":"abc-123","timestamp":"2026-07-01T10:00:05Z","message":{"role":"assistant","content":[{"type":"text","text":"Looking into it."},{"type":"tool_use","name":"Bash","input":{"command":"cargo test"}}]}}
{"type":"user","sessionId":"abc-123","message":{"role":"user","content":[{"type":"tool_result","content":"test output"}]}}"#;
        let transcript = parse_transcript(content, "fallback", "test.jsonl").unwrap();
        assert_eq!(transcript.id, "abc-123");
        assert_eq!(transcript.messages.len(), 2);
        assert_eq!(transcript.messages[0].role, "user");
        assert_eq!(transcript.messages[0].content, "Please fix the flaky test");
        assert!(transcript.messages[1]
            .content
            .contains("ToolCall: name=Bash; input=command=cargo test"));
    }

    #[test]
    fn parses_generic_format() {
        let content = r#"{"role":"user","content":"I prefer tabs over spaces"}
{"role":"assistant","content":"Noted."}"#;
        let transcript = parse_transcript(content, "generic-1", "chat.jsonl").unwrap();
        assert_eq!(transcript.id, "generic-1");
        assert_eq!(transcript.messages.len(), 2);
    }

    #[test]
    fn skips_meta_and_machine_lines() {
        let content = r#"{"type":"user","isMeta":true,"message":{"role":"user","content":"meta"}}
{"type":"user","message":{"role":"user","content":"<command-name>/clear</command-name>"}}
{"type":"user","message":{"role":"user","content":"real question"}}"#;
        let transcript = parse_transcript(content, "s", "f.jsonl").unwrap();
        assert_eq!(transcript.messages.len(), 1);
        assert_eq!(transcript.messages[0].content, "real question");
    }

    #[test]
    fn rejects_non_jsonl_content() {
        assert!(parse_transcript("not json at all", "s", "f.txt").is_err());
    }
}
