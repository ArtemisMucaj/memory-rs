//! Prompt construction for memory ingestion.
//!
//! One bounded LLM call per session: given the conversation and a handful of
//! prior memories prefetched by semantic similarity, the model emits atomic
//! facts plus the entity mentions that anchor them.
//!
//! The schema is deliberately flat. The online tier runs a small local model
//! and not every chat backend supports structured output, so nested or
//! polymorphic schemas measurably raise the malformed-output rate.
//!
//! There is no `predicate` and no `subject`/`object` split — the verb is in
//! the statement, where a reader looks for it. Entities are listed as
//! *mentions* and resolved server-side by name-key.

use crate::domain::{Memory, SessionTranscript, VALID_ENTITY_TYPES};

/// Maximum characters of conversation text sent to the extraction model.
///
/// Longer transcripts keep the head *and* tail and elide the middle. Durable
/// preferences cluster at session *start* (where the user says what they
/// want) and outcomes cluster at the end; a tail-only window silently loses
/// the first.
pub const MAX_CONVERSATION_CHARS: usize = 60_000;

/// Maximum characters quoted per prior memory in the prompt.
const MAX_PRIOR_MEMORY_CHARS: usize = 400;

/// System prompt: what a fact is, and how to mark entity mentions.
pub fn system_prompt() -> String {
    let entity_types = VALID_ENTITY_TYPES
        .iter()
        .map(|t| format!("`{t}`"))
        .collect::<Vec<_>>()
        .join(", ");
    r#"You extract what is worth remembering from a finished coding-assistant session, as atomic FACTS.

A fact is ONE self-contained English sentence worth remembering across sessions. If a sentence contains two facts, emit two memories.

## Memory shape

- `statement` — the fact as a single sentence. This is what gets embedded and shown to a reader, so it must read on its own. Include the name of the thing the fact is about, in the form it is written down ("orders-events", never "the orders-events service").
- `source_kind` — `user_stated` if the USER asserted it directly, `extracted` for anything you produce from the transcript (whether read straight off the page or pieced together from context).
- `confidence` — 0..1.
- `entity_mentions` — the durable things this fact is *about*. See "Entities" below. Each mention is `{ "name": "...", "type": "..." }`. An empty array is fine — many facts are about the user or about nothing durable.

## Entities

An entity is a **durable thing that recurs across sessions** — a person, a tool, a service, a library, a standing concept. It is an *anchor*: every fact about the same thing must mention the same entity.

Apply this test: **would this still be referred to by name in six months, in a different conversation?**

These are **not** entities. Leave them out of `entity_mentions` and let them live inside the statement as plain text:
- a file, path, directory, function, class, variable or symbol
- a commit, branch, pull request or ticket
- a version number, error message, log line or environment variable
- anything that exists only inside the change you are discussing

If a fact's natural subject is a specific file or symbol, the fact is almost always really about the **service, library or tool that file belongs to**. Mention that one instead.

**Name an entity the way it is written down.** Strip the article and the role word — write `orders-events`, never "the orders-events service" or "orders-events package". Put the role in the statement, where it belongs.

### Picking the type

{{ENTITY_TYPES}}, or `unknown` when genuinely unclear.

- `person` — a human being.
- `tool` — third-party software: Kafka, Terraform, Postgres, DuckDB, a language, a framework, a CLI.
- `service` — a running piece of the user's own system: an API, a worker, a daemon.
- `library` — a reusable piece of the user's own codebase: a crate, a package, a module.
- `concept` — a standing idea with no artefact behind it.
- `unknown` — genuinely unclear. Prefer it to a guess.

**Projects are not entities.** The repository the session ran in is recorded on the memory itself, not as an entity. Do not mention the project name as an entity.

## The bar for emitting anything

The bar is HIGH. Prefer FEWER, higher-value facts. An empty list is better than noise, and is the correct answer for many sessions.

Before emitting a fact, apply the "still useful in 3 months?" test. If it is a snapshot of what this session did, DROP it:
- DO store: an architectural decision and why; a stable tooling choice; a durable user habit.
- Do NOT store: a version number being bumped to, which files a change touched, "the current failure is caused by X", or any in-flight state. These are session logs, not durable memories.

User-authored messages are the source of truth for facts about the user; assistant and tool activity is the source for facts about the system. Never invent anything the transcript does not support.

## Output

Respond with ONLY a JSON object — no prose, no markdown fence:

{"memories": [{"statement": "...", "source_kind": "user_stated", "confidence": 0.9, "entity_mentions": [{"name": "orders-events", "type": "service"}]}]}

Return `{"memories": []}` when the session contains nothing worth remembering."#
        .to_string()
        .replace("{{ENTITY_TYPES}}", &entity_types)
}

/// User prompt: prior memories followed by the conversation.
pub fn user_prompt(transcript: &SessionTranscript, prior: &[Memory]) -> String {
    let mut out = String::new();
    if prior.is_empty() {
        out.push_str("PRIOR MEMORIES: (none)\n\n");
    } else {
        out.push_str("PRIOR MEMORIES:\n");
        for memory in prior {
            out.push_str(&format!(
                "- {}\n",
                truncate_chars(memory.statement.trim(), MAX_PRIOR_MEMORY_CHARS),
            ));
        }
        out.push('\n');
    }
    out.push_str("CONVERSATION:\n");
    out.push_str(&render_conversation(transcript));
    out.push_str(
        "\n\nExtract the durable facts from this conversation as a single JSON object. \
         Output nothing except the JSON object.",
    );
    out
}

/// Retry message appended after an unparseable response.
pub fn format_retry_prompt() -> &'static str {
    "Your previous output could not be parsed as valid JSON. Output ONLY a valid JSON object \
     with a single field `memories` holding an array. Do not include any explanation, markdown \
     formatting, or text outside the JSON."
}

/// Render `[idx][role]: content` lines, eliding the middle of transcripts
/// that exceed [`MAX_CONVERSATION_CHARS`].
fn render_conversation(transcript: &SessionTranscript) -> String {
    let max_per_message = MAX_CONVERSATION_CHARS / 3;
    let lines: Vec<String> = transcript
        .messages
        .iter()
        .enumerate()
        .filter(|(_, m)| !m.content.trim().is_empty())
        .map(|(idx, m)| {
            let content = truncate_chars(m.content.trim(), max_per_message);
            format!("[{}][{}]: {}", idx, m.role, content)
        })
        .collect();

    let total: usize = lines.iter().map(|l| l.len() + 2).sum();
    if total <= MAX_CONVERSATION_CHARS {
        return lines.join("\n\n");
    }

    let head_budget = MAX_CONVERSATION_CHARS / 2;
    let tail_budget = MAX_CONVERSATION_CHARS - head_budget;

    let mut head: Vec<&String> = Vec::new();
    let mut used = 0usize;
    for line in &lines {
        if used + line.len() > head_budget {
            break;
        }
        used += line.len() + 2;
        head.push(line);
    }

    let mut tail: Vec<&String> = Vec::new();
    used = 0;
    for line in lines.iter().rev() {
        if used + line.len() > tail_budget {
            break;
        }
        used += line.len() + 2;
        tail.push(line);
    }
    tail.reverse();

    let elided = lines.len().saturating_sub(head.len() + tail.len());
    let mut out: Vec<&str> = head.iter().map(|s| s.as_str()).collect();
    let marker = format!("[... {elided} messages elided ...]");
    if elided > 0 {
        out.push(&marker);
    }
    out.extend(tail.iter().map(|s| s.as_str()));
    out.join("\n\n")
}

/// Build a compact semantic query from the transcript for prefetching
/// related prior memories: user messages first, assistant text as
/// supporting signal.
pub fn prefetch_query(transcript: &SessionTranscript) -> String {
    const MAX_QUERY_CHARS: usize = 4_000;
    const USER_PART_CHARS: usize = 800;
    const ASSISTANT_PART_CHARS: usize = 300;

    let mut primary = Vec::new();
    let mut supporting = Vec::new();
    for msg in &transcript.messages {
        let text = msg.content.trim();
        if text.is_empty() {
            continue;
        }
        if msg.role == "user" {
            primary.push(truncate_chars(text, USER_PART_CHARS));
        } else {
            supporting.push(truncate_chars(text, ASSISTANT_PART_CHARS));
        }
    }
    primary.extend(supporting);
    truncate_chars(&primary.join("\n"), MAX_QUERY_CHARS)
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let truncated: String = text.chars().take(max_chars.saturating_sub(3)).collect();
    format!("{truncated}...")
}

/// JSON Schema for the extraction output, kept flat for structured-output
/// backends. Mirrors `RawIngestion` / `RawMemory` in
/// [`memory_ingestion`](super::memory_ingestion).
pub fn schema() -> serde_json::Value {
    let entity_types: Vec<&str> = VALID_ENTITY_TYPES
        .iter()
        .copied()
        .chain(std::iter::once("unknown"))
        .collect();
    serde_json::json!({
        "type": "object",
        "properties": {
            "memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "statement": { "type": "string" },
                        "source_kind": {
                            "type": "string",
                            "enum": ["user_stated", "extracted"]
                        },
                        "confidence": { "type": "number" },
                        "entity_mentions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": { "type": "string" },
                                    "type": { "type": "string", "enum": entity_types }
                                },
                                "required": ["name", "type"],
                                "additionalProperties": false
                            }
                        }
                    },
                    "required": ["statement", "source_kind", "confidence"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["memories"],
        "additionalProperties": false
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::SessionMessage;

    fn message(role: &str, content: &str) -> SessionMessage {
        SessionMessage {
            role: role.to_string(),
            content: content.to_string(),
            timestamp: None,
        }
    }

    fn transcript(messages: Vec<SessionMessage>) -> SessionTranscript {
        SessionTranscript {
            id: "session-1".to_string(),
            source: "test".to_string(),
            project: Some("owner/repo".to_string()),
            messages,
        }
    }

    #[test]
    fn short_conversation_is_rendered_whole() {
        let t = transcript(vec![
            message("user", "use tabs"),
            message("assistant", "noted"),
        ]);
        let rendered = render_conversation(&t);
        assert!(rendered.contains("[0][user]: use tabs"));
        assert!(rendered.contains("[1][assistant]: noted"));
        assert!(!rendered.contains("elided"));
    }

    #[test]
    fn long_conversation_keeps_head_and_tail_and_elides_the_middle() {
        let filler = "x".repeat(5_000);
        let mut messages = vec![message("user", "REMEMBER THE OPENING")];
        for _ in 0..40 {
            messages.push(message("assistant", &filler));
        }
        messages.push(message("user", "REMEMBER THE ENDING"));
        let rendered = render_conversation(&transcript(messages));
        assert!(rendered.contains("REMEMBER THE OPENING"));
        assert!(rendered.contains("REMEMBER THE ENDING"));
        assert!(rendered.contains("messages elided"));
    }

    #[test]
    fn prefetch_query_puts_user_messages_first() {
        let t = transcript(vec![
            message("assistant", "assistant text"),
            message("user", "user text"),
        ]);
        let query = prefetch_query(&t);
        let user_at = query.find("user text").expect("user text missing");
        let assistant_at = query
            .find("assistant text")
            .expect("assistant text missing");
        assert!(user_at < assistant_at);
    }

    #[test]
    fn schema_does_not_ask_for_kind_predicate_or_relation() {
        let schema = schema();
        let props = schema["properties"]["memories"]["items"]["properties"]
            .as_object()
            .unwrap();
        assert!(!props.contains_key("kind"));
        assert!(!props.contains_key("predicate"));
        assert!(!props.contains_key("relation"));
        assert!(!props.contains_key("subject"));
        assert!(!props.contains_key("object"));
        // The fields the new shape actually reads.
        for field in ["statement", "source_kind", "confidence", "entity_mentions"] {
            assert!(props.contains_key(field), "schema missing '{field}'");
        }
    }

    #[test]
    fn prompt_does_not_list_project_as_an_entity_type() {
        let p = system_prompt();
        assert!(!p.contains("`project`"));
    }
}
