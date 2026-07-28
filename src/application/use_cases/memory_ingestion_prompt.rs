//! Prompt construction for memory ingestion.
//!
//! One bounded LLM call per session: given the conversation and a handful of
//! prior memories (with ids) prefetched by semantic similarity, the model emits
//! atomic subject–predicate–object memories and, optionally, a typed relation
//! from each new memory to one of those priors.
//!
//! The schema is deliberately flat. The online tier runs a small local model
//! and not every chat backend supports structured output, so nested or
//! polymorphic schemas measurably raise the malformed-output rate. Every field
//! is a scalar or a one-level object.
//!
//! The anti-noise rules here are inherited from the item-extraction prompt
//! rather than reinvented, because they were tuned against real transcripts and
//! the failure they prevent — a store full of session logs nobody will ever
//! want — is identical under memories. What changes is that they now apply to
//! *atomic* memories, which are individually smaller and therefore easier for a
//! model to emit thoughtlessly in bulk.

use crate::domain::{Memory, MemoryKind, SessionTranscript};

/// Maximum characters of conversation text sent to the extraction model.
///
/// Longer transcripts keep the head *and* tail and elide the middle. This is
/// the single most important departure from a naive "keep the last N chars"
/// window: durable preferences cluster at session *start*, where the user says
/// what they want, while outcomes cluster at the end. A tail-only window sees
/// the second and silently loses the first.
pub const MAX_CONVERSATION_CHARS: usize = 60_000;

/// Maximum characters quoted per prior memory in the prompt. Memories are short by
/// construction, so this only bites on a pathological statement.
const MAX_PRIOR_MEMORY_CHARS: usize = 400;

/// System prompt: what a memory is, which kinds exist, and when to relate a new
/// memory to a prior one.
pub fn system_prompt() -> String {
    r#"You extract what is worth remembering from a finished coding-assistant session, as atomic MEMORIES.

A memory is ONE subject–predicate–object fact worth remembering across sessions. Keep it atomic: one fact per memory. If a sentence contains two facts, emit two memories.

## Memory shape

- `subject` — what the memory is about (usually a person, project, or tool). `subject_is_entity` is almost always true.
- `subject_type` — one of: person, project, tool, place, organization, concept, unknown. Use `unknown` only when genuinely unclear; a wrong guess is worse than `unknown`, because entities of differing types are never merged.
- `predicate` — a short snake_case relation: prefers, uses, decided, lives_in, requires, avoids.
- `object` — the value. Set `object_is_entity` true when it names a distinct entity (a project, tool, person), false when it is a literal value ("tabs", "4", "2025-01-01").
- `object_type` — same vocabulary as `subject_type`. Ignored when `object_is_entity` is false.
- `statement` — a short, self-contained sentence rendering the triple. This is what gets embedded and shown to a human, so it must read on its own without the subject/predicate/object fields.
- `kind` — see below.
- `source_kind` — `user_stated` if the USER asserted it directly, `assistant_inferred` if you concluded it from context. Be honest: this field decides who wins a future contradiction, and inflating it lets a guess overwrite something the user actually said.
- `confidence` — 0..1.

## Kinds

Every memory is EXACTLY ONE kind. The same insight must never appear under two kinds.

- `preference` — a durable taste or habit of the USER ("prefers tabs over spaces", "wants tool-call arguments shown"). A LASTING habit that holds across many future sessions, not a one-off goal for this session. "The user is upgrading X to v2" is a goal, not a preference.
- `fact` — a durable, declarative truth about the PROJECT or environment, WITH its rationale where known ("logging goes to stderr in MCP mode because stdout carries the protocol").
- `experience` — a reusable lesson about how something breaks and how it was fixed. Generalize it: strip specific ids, paths and raw text so the lesson applies beyond the instance that produced it.
- `skill` — a repeatable multi-step PROCEDURE an agent would run again from scratch, independent of any one bug (a release flow, a debugging recipe).

The two boundaries that get confused most:
- preference vs fact — a statement about what the USER likes is a preference; a statement about how the CODE is built is a fact. "The user set the default model to X" is a preference; "the project's default model is X" is a fact. Pick ONE, never both.
- experience vs skill — a one-off fix or debugging lesson is an EXPERIENCE only. A generic procedure you would run again from scratch is a SKILL only. Implementing a feature once is an experience. If in doubt, it is an experience.

## The bar for emitting anything

The bar is HIGH. Prefer FEWER, higher-value memories. An empty list is better than noise, and is the correct answer for many sessions.

Before emitting a memory, apply the "still useful in 3 months?" test. If it is a snapshot of what this session did, DROP it:
- DO store: an architectural decision and why; a stable tooling choice; a durable user habit.
- Do NOT store: a version number being bumped to, which files a change touched, "the current failure is caused by X", or any in-flight state. These are session logs, not durable memories.

User-authored messages are the source of truth for preferences and for facts about the user; assistant and tool activity is the source for experiences and skills. Never invent anything the transcript does not support.

## Relating to prior memories

You are given PRIOR MEMORIES with ids. When a new memory bears on one of them, set `relation` to relate the NEW memory to that prior id:

- `supersedes` — the new memory replaces an out-of-date prior memory. Use this only for genuine change over time ("moved from Munich to Berlin"), not for a rephrasing.
- `refines` — the new memory is a more specific version of the prior one, and BOTH stay true.
- `contradicts` — they genuinely conflict and you cannot tell which is right.
- `corroborates` — the new memory independently confirms the prior one.

Only set `relation` when you are confident about the target id; omit it otherwise. A relation naming an id that is not in the PRIOR MEMORIES list is discarded.

## Output

Respond with ONLY a JSON object — no prose, no markdown fence:

{"memories": [{"subject": "...", "subject_is_entity": true, "subject_type": "person", "predicate": "...", "object": "...", "object_is_entity": false, "object_type": "unknown", "statement": "...", "kind": "preference", "source_kind": "user_stated", "confidence": 0.9}]}

Return `{"memories": []}` when the session contains nothing worth remembering."#
        .to_string()
}

/// User prompt: prior memories (with ids) followed by the conversation.
pub fn user_prompt(transcript: &SessionTranscript, prior: &[Memory]) -> String {
    let mut out = String::new();
    if prior.is_empty() {
        out.push_str("PRIOR MEMORIES: (none)\n\n");
    } else {
        out.push_str("PRIOR MEMORIES (id — [kind] statement):\n");
        for memory in prior {
            out.push_str(&format!(
                "- {} — [{}] {}\n",
                memory.id,
                memory.kind.as_str(),
                truncate_chars(memory.statement.trim(), MAX_PRIOR_MEMORY_CHARS),
            ));
        }
        out.push('\n');
    }
    out.push_str("CONVERSATION:\n");
    out.push_str(&render_conversation(transcript));
    out.push_str(
        "\n\nExtract the durable memories from this conversation as a single JSON object. \
         Output nothing except the JSON object.",
    );
    out
}

/// Retry message appended after an unparseable response, giving the model one
/// chance to correct its output format.
pub fn format_retry_prompt() -> &'static str {
    "Your previous output could not be parsed as valid JSON. Output ONLY a valid JSON object \
     with a single field `memories` holding an array. Do not include any explanation, markdown \
     formatting, or text outside the JSON."
}

/// Render `[idx][role]: content` lines, eliding the middle of transcripts that
/// exceed [`MAX_CONVERSATION_CHARS`].
///
/// Two caps, for two different failure modes. The per-message cap stops one
/// oversized message — a pasted file or a stack trace — from either eating the
/// whole budget or being dropped whole. The head/tail split then keeps entire
/// messages from both ends, because that is where durable signal lives.
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

/// Build a compact semantic query from the transcript for prefetching related
/// prior memories: user messages first, assistant text as supporting signal.
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
/// backends. Mirrors the `RawIngestion` / `RawMemory` / `RawRelation` structs in
/// [`memory_ingestion`](super::memory_ingestion).
pub fn schema() -> serde_json::Value {
    let entity_types = [
        "person",
        "project",
        "tool",
        "place",
        "organization",
        "concept",
        "unknown",
    ];
    let kinds: Vec<&str> = MemoryKind::ALL.iter().map(|k| k.as_str()).collect();
    serde_json::json!({
        "type": "object",
        "properties": {
            "memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": { "type": "string" },
                        "subject_is_entity": { "type": "boolean" },
                        "subject_type": { "type": "string", "enum": entity_types },
                        "predicate": { "type": "string" },
                        "object": { "type": "string" },
                        "object_is_entity": { "type": "boolean" },
                        "object_type": { "type": "string", "enum": entity_types },
                        "statement": { "type": "string" },
                        "kind": { "type": "string", "enum": kinds },
                        "source_kind": {
                            "type": "string",
                            "enum": ["user_stated", "assistant_inferred"]
                        },
                        "confidence": { "type": "number" },
                        "relation": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["supersedes", "refines", "contradicts", "corroborates"]
                                },
                                "target": { "type": "string" }
                            },
                            "required": ["type", "target"],
                            "additionalProperties": false
                        }
                    },
                    "required": [
                        "subject", "subject_is_entity", "predicate", "object",
                        "object_is_entity", "statement", "kind", "source_kind", "confidence"
                    ],
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
    fn empty_messages_are_skipped_but_do_not_renumber_the_rest() {
        // The index is the message's real position in the transcript, so a
        // blank message leaves a gap. Renumbering would make the indices lie
        // about which message a memory came from.
        let t = transcript(vec![
            message("user", "first"),
            message("assistant", "   "),
            message("user", "third"),
        ]);
        let rendered = render_conversation(&t);
        assert!(rendered.contains("[0][user]: first"));
        assert!(rendered.contains("[2][user]: third"));
        assert!(!rendered.contains("[1]"));
    }

    /// The failure this guards: a tail-only window (what a naive port does)
    /// keeps the end of the session and silently loses the beginning — which is
    /// exactly where a user states the preferences worth remembering.
    #[test]
    fn long_conversation_keeps_head_and_tail_and_elides_the_middle() {
        let filler = "x".repeat(5_000);
        let mut messages = vec![message("user", "REMEMBER THE OPENING")];
        for _ in 0..40 {
            messages.push(message("assistant", &filler));
        }
        messages.push(message("user", "REMEMBER THE ENDING"));
        let rendered = render_conversation(&transcript(messages));

        assert!(
            rendered.contains("REMEMBER THE OPENING"),
            "head was dropped — durable preferences cluster at session start",
        );
        assert!(rendered.contains("REMEMBER THE ENDING"), "tail was dropped",);
        assert!(rendered.contains("messages elided"));
        assert!(rendered.len() <= MAX_CONVERSATION_CHARS + 1_000);
    }

    /// One pasted file must not consume the whole budget, and must not be
    /// dropped whole either — it gets truncated so its neighbours still fit.
    #[test]
    fn one_oversized_message_is_capped_rather_than_dropped() {
        let huge = "y".repeat(MAX_CONVERSATION_CHARS * 2);
        let t = transcript(vec![
            message("user", "before"),
            message("assistant", &huge),
            message("user", "after"),
        ]);
        let rendered = render_conversation(&t);
        assert!(rendered.contains("before"));
        assert!(rendered.contains("after"));
        assert!(
            rendered.contains("..."),
            "the huge message was not truncated"
        );
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
        assert!(
            user_at < assistant_at,
            "user messages must lead the prefetch query — they carry the durable signal",
        );
    }

    #[test]
    fn user_prompt_lists_prior_memory_ids_and_kinds() {
        let prior = vec![Memory {
            id: "memory-1".to_string(),
            kind: MemoryKind::Preference,
            subject: crate::domain::EntityRef::Entity("entity-1".to_string()),
            predicate: "prefers".to_string(),
            object: crate::domain::EntityRef::Literal("tabs".to_string()),
            statement: "the user prefers tabs".to_string(),
            project: None,
            recorded_at: 1,
            valid_from: 1,
            valid_to: None,
            source_session_id: None,
            source_message_index: None,
            source_kind: crate::domain::SourceKind::UserStated,
            confidence: 0.9,
            status: crate::domain::MemoryStatus::Active,
            derived: false,
            derived_from: Vec::new(),
        }];
        let rendered = user_prompt(&transcript(vec![message("user", "hi")]), &prior);
        assert!(rendered.contains("memory-1 — [preference] the user prefers tabs"));

        let none = user_prompt(&transcript(vec![message("user", "hi")]), &[]);
        assert!(none.contains("PRIOR MEMORIES: (none)"));
    }

    #[test]
    fn schema_declares_the_fields_the_parser_reads() {
        let schema = schema();
        let props = schema["properties"]["memories"]["items"]["properties"]
            .as_object()
            .unwrap();
        for field in [
            "subject",
            "subject_is_entity",
            "subject_type",
            "predicate",
            "object",
            "object_is_entity",
            "object_type",
            "statement",
            "kind",
            "source_kind",
            "confidence",
            "relation",
        ] {
            assert!(props.contains_key(field), "schema missing '{field}'");
        }
        // `kind` must offer exactly the domain's kinds — a schema listing a
        // kind the parser cannot map would silently fall back to `fact`.
        let kinds: Vec<&str> = schema["properties"]["memories"]["items"]["properties"]["kind"]
            ["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(kinds, ["preference", "experience", "skill", "fact"]);
    }
}
