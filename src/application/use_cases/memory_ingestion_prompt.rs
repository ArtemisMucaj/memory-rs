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

use crate::domain::{Memory, MemoryKind, Predicate, SessionTranscript};

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
    let relations = Predicate::ALL
        .iter()
        .map(|p| format!("- `{}` — {}", p.as_str(), p.meaning()))
        .collect::<Vec<_>>()
        .join("\n");
    r#"You extract what is worth remembering from a finished coding-assistant session, as atomic MEMORIES.

A memory is ONE subject–predicate–object fact worth remembering across sessions. Keep it atomic: one fact per memory. If a sentence contains two facts, emit two memories.

## Memory shape

- `subject` — what the memory is about. Set `subject_is_entity` true only when it is an entity by the test under "Entities" below.
- `subject_type` — one of: person, project, tool, place, organization, concept, unknown. Use `unknown` only when genuinely unclear; a wrong guess is worse than `unknown`, because entities of differing types are never merged.
- `predicate` — the relation, chosen from the CLOSED list under "Relations" below. You must use one of those exact values and nothing else.
- `object` — the value. Set `object_is_entity` true only when it names a distinct entity by that same test; false for a literal value ("tabs", "4", "2025-01-01", a file path, an error message).
- `object_type` — same vocabulary as `subject_type`. Ignored when `object_is_entity` is false.
- `statement` — a short, self-contained sentence rendering the triple. This is what gets embedded and shown to a human, so it must read on its own without the subject/predicate/object fields.
- `kind` — see below.
- `source_kind` — `user_stated` if the USER asserted it directly, `assistant_inferred` if you concluded it from context. Be honest: this field decides who wins a future contradiction, and inflating it lets a guess overwrite something the user actually said.
- `confidence` — 0..1.

## Entities

An entity is a **durable thing that recurs across sessions** — a person, a repository or service, a tool, a team, a place, a standing concept. It is an *anchor*: every memory about the same thing must point at the same entity, and that is the only reason the store is a graph rather than a list.

Apply this test: **would this still be referred to by name in six months, in a different conversation?**

These are **not** entities. Mark them `is_entity: false` and let them be plain values:
- a file, path, directory, function, class, variable or symbol
- a commit, branch, pull request or ticket
- a version number, error message, log line or environment variable
- anything that exists only inside the change you are discussing

If a memory's natural subject is a specific file or symbol, the memory is almost always really about the **component, service or repository that file belongs to**. Use that as the subject and put the specific detail in the statement. "src/auth/token.rs validates JWTs" is better recorded as the auth service validating JWTs.

Getting this wrong is expensive in a way that is not obvious: a file promoted to an entity becomes a permanent anchor that nothing will ever reference again, and it is typed wrongly too — there is no file type in the list, so it lands as `project` and blocks that name from ever merging with the real project.

**Name an entity the way it is written down**, not the way the sentence refers to it: the repository name, the package name, the person's name. Strip the article and the role word — write `orders-events`, never "the orders-events service" or "orders-events package". Put the role in the statement, where it belongs ("orders-events is the service that…"). Two spellings of one name become two anchors, and nothing later can tell they were the same thing.

### Picking the type

The type is part of an entity's identity — two entities typed differently are **never** merged — so a careless type is as permanent as a careless name. The line that matters is who owns the thing:

- `project` — something the user works ON: their repository, service, package or app. If it lives in their codebase, it is a project.
- `tool` — third-party software they work WITH: Kafka, Terraform, Postgres, DuckDB, Docker, a language, a framework, a CLI. You do not own it, you use it.
- `person`, `organization`, `place` — as they sound. A team is an organization.
- `concept` — a standing idea with no artefact behind it ("event sourcing", "the release process").
- `unknown` — genuinely unclear. Prefer it to a guess.

Typing a tool as a project is the common mistake, and it costs twice: it is wrong, and the next session that types it correctly gets a second anchor for the same thing.

## Relations

`predicate` must be **exactly one** of these. Pick on meaning, not on which word looks closest to the sentence — the relation is half of how two memories are recognised as the same fact, so an invented synonym silently stores the same thing twice.

{{RELATIONS}}

Before reaching for `relates_to`, **read the list again**. It is the last resort, not the default — a memory that lands there is one the store cannot group with anything, so it is close to useless. In practice it is almost always wrong:

- "X involves A and B", "X consists of A and B", "X has parts A and B" → `contains`
- "X was built to mirror Y", "X is based on Y" → `derived_from`
- "X needs Y to work" → `requires`
- "X sets up Y" → `configures`

Use `relates_to` only when you have checked every entry and none of them expresses the relation. Do not invent a relation and do not stretch one that nearly fits.

## Kinds

Every memory is EXACTLY ONE kind. The same insight must never appear under two kinds.

- `preference` — a durable taste or habit of the USER ("prefers tabs over spaces", "wants tool-call arguments shown"). A LASTING habit that holds across many future sessions, not a one-off goal for this session. "The user is upgrading X to v2" is a goal, not a preference.
- `fact` — a durable, declarative truth about the PROJECT or environment, WITH its rationale where known ("logging goes to stderr in MCP mode because stdout carries the protocol").
- `experience` — a reusable lesson about how something breaks and how it was fixed. Generalize it: strip specific ids, paths and raw text so the lesson applies beyond the instance that produced it.
- `skill` — INSTRUCTIONS someone could follow. The statement must tell a reader what to DO, in order: "to cut a release, bump the version, merge the release PR, then verify the tag". A release flow, a debugging recipe, a setup procedure.

`skill` is the kind that gets over-used, so apply this test before choosing it: **could a reader follow the statement and perform the procedure?** If the statement merely *describes* something — what a system does, what a component is responsible for, how a pipeline is wired — it is a FACT, no matter how many steps it mentions.

- "The X service reads the queue, validates each message, writes it to the database and emits a metric" — describes what X does. That is a **fact**.
- "To add a message type, register it in the catalog, add a decoder, then run the codegen" — tells you what to do. That is a **skill**.

A statement whose subject is a system and whose verb is "implements", "handles", "does" or "consists of" is a fact about that system. Naming a procedure is not the same as giving one.

The boundaries that get confused most:
- preference vs fact — a statement about what the USER likes is a preference; a statement about how the CODE is built is a fact. "The user set the default model to X" is a preference; "the project's default model is X" is a fact. Pick ONE, never both.
- experience vs skill — a one-off fix or debugging lesson is an EXPERIENCE only. A generic procedure you would run again from scratch is a SKILL only. Implementing a feature once is an experience. If in doubt, it is an experience.
- fact vs skill — describing a system is a FACT; instructing a reader is a SKILL. If in doubt, it is a fact.

## The bar for emitting anything

The bar is HIGH. Prefer FEWER, higher-value memories. An empty list is better than noise, and is the correct answer for many sessions.

Before emitting a memory, apply the "still useful in 3 months?" test. If it is a snapshot of what this session did, DROP it:
- DO store: an architectural decision and why; a stable tooling choice; a durable user habit.
- Do NOT store: a version number being bumped to, which files a change touched, "the current failure is caused by X", or any in-flight state. These are session logs, not durable memories.
- In particular, an edit to a document or file — "the section about X was removed", "a paragraph was added" — is a change record, not a fact. The durable version, if there is one, is what the code or the decision now *is*.

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

{"memories": [{"subject": "...", "subject_is_entity": true, "subject_type": "person", "predicate": "uses", "object": "...", "object_is_entity": false, "object_type": "unknown", "statement": "...", "kind": "preference", "source_kind": "user_stated", "confidence": 0.9}]}

Return `{"memories": []}` when the session contains nothing worth remembering."#
        .to_string()
        .replace("{{RELATIONS}}", &relations)
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

/// System prompt for entity adjudication — the tier that runs *only* in the
/// ambiguous similarity band, where two names are too close to call apart and
/// too far to merge outright.
pub fn entity_adjudication_system_prompt() -> String {
    // The "package vs the service that consumes it" example this used to give
    // was the exact shape of the case it most needed to get right: shown
    // "orders-events package" and "orders-events service" — one service,
    // written down twice — a small model matched the example and answered
    // false, minting a permanent duplicate anchor. Same base name is now
    // called out as evidence *for* sameness; the distinctness examples are the
    // ones where the names themselves differ.
    "You decide whether two names refer to the same thing.\n\n\
     You are given two names that a similarity search found close but not \
     identical, with the kind of thing each is believed to be and an example of \
     how each was used.\n\n\
     Answer `true` if a reader would consider them the *same* thing under two \
     names — a service and its abbreviation, a repository and its full path, a \
     person and their handle. In particular, answer `true` when the names share \
     a base and differ only by a describing word (`orders` / `orders service` / \
     `the orders package` / `orders repo`): that is one thing written down two \
     ways, which is the most common way a duplicate is created.\n\n\
     Answer `false` when the names themselves differ — two siblings sharing a \
     prefix (`auth-api` and `auth-worker`), a project and a component of it that \
     has its own name, two people with the same first name. Being about the same \
     area is not being the same thing.\n\n\
     When genuinely unsure, answer `false`. Merging two distinct entities cannot \
     be undone — every memory anchored to either one silently becomes a memory \
     about a thing that does not exist.\n\n\
     Respond with ONLY a JSON object: {\"same\": true|false}"
        .to_string()
}

/// User prompt for one adjudication.
pub fn entity_adjudication_user_prompt(
    a_name: &str,
    a_type: &str,
    a_example: &str,
    b_name: &str,
    b_type: &str,
    b_example: &str,
) -> String {
    format!(
        "A: {a_name}\n  kind: {a_type}\n  seen in: {a_example}\n\n         B: {b_name}\n  kind: {b_type}\n  seen in: {b_example}\n\n         Are A and B the same thing?"
    )
}

/// JSON Schema for the adjudication answer.
pub fn entity_adjudication_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": { "same": { "type": "boolean" } },
        "required": ["same"],
        "additionalProperties": false
    })
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
                        "predicate": {
                            "type": "string",
                            "enum": Predicate::ALL.iter().map(|p| p.as_str()).collect::<Vec<_>>()
                        },
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
            predicate: Predicate::Prefers,
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

    /// The prompt and the schema must offer the *same* closed set. If they ever
    /// drift, a structured-output backend rejects a value the prompt asked for
    /// — or worse, a lenient backend lets through a value the prompt never
    /// mentioned and every one of those memories lands on `relates_to`.
    #[test]
    fn relations_block_lists_every_predicate_and_matches_the_schema() {
        let prompt = system_prompt();
        for p in Predicate::ALL {
            assert!(
                prompt.contains(&format!("`{}`", p.as_str())),
                "predicate '{}' missing from the prompt's Relations list",
                p.as_str(),
            );
            assert!(
                prompt.contains(p.meaning()),
                "predicate '{}' has no meaning in the prompt",
                p.as_str(),
            );
        }
        assert!(
            !prompt.contains("{{RELATIONS}}"),
            "placeholder not substituted"
        );

        let schema = schema();
        let enumerated: Vec<&str> = schema["properties"]["memories"]["items"]["properties"]
            ["predicate"]["enum"]
            .as_array()
            .expect("predicate must be an enum in the schema")
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let expected: Vec<&str> = Predicate::ALL.iter().map(|p| p.as_str()).collect();
        assert_eq!(enumerated, expected);
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

#[cfg(test)]
mod entity_guidance {
    use super::*;
    /// The failure this guards: the prompt used to say `subject_is_entity` is
    /// "almost always true", which promoted files and symbols to permanent
    /// anchors that nothing ever references again — and, with no file type in
    /// the list, typed them `project`, blocking the real project from merging.
    #[test]
    fn the_prompt_says_what_is_not_an_entity() {
        let p = system_prompt();
        assert!(p.contains("## Entities"));
        assert!(!p.contains("almost always true"));
        for excluded in ["file", "commit", "version number", "symbol"] {
            assert!(p.contains(excluded), "entity guidance omits {excluded:?}");
        }
    }

    /// The failure this guards: one session named the same service "the
    /// orders-events service" once and "orders-events package" once, and the
    /// store ended up with two anchors for it. Normalization catches most of
    /// that downstream, but only the prompt can stop the role word being
    /// written into the name in the first place.
    #[test]
    fn the_prompt_tells_the_model_to_drop_role_words_from_names() {
        let p = system_prompt();
        assert!(p.contains("Strip the article and the role word"));
    }

    /// The failure this guards: a statement describing what a service does was
    /// stored as a `skill`, which is meant to be a procedure an agent can
    /// follow. The kind list alone did not discriminate — the describing
    /// statement mentioned several steps, so it looked procedural.
    #[test]
    fn the_prompt_separates_describing_a_system_from_instructing_a_reader() {
        let p = system_prompt();
        assert!(p.contains("could a reader follow the statement and perform the procedure?"));
        assert!(p.contains("fact vs skill"));
    }

    /// The failure this guards: `Kafka` and `Terraform` were both stored as
    /// `project`. The field listed its vocabulary but never said what the words
    /// meant, so everything software-shaped landed on the first plausible one.
    /// A wrong type is permanent — the guard that keeps a `tool` and a
    /// `project` apart is the same one that stops the correction merging.
    #[test]
    fn the_prompt_says_what_each_entity_type_means() {
        let p = system_prompt();
        assert!(p.contains("### Picking the type"));
        assert!(p.contains("something the user works ON"));
        assert!(p.contains("third-party software they work WITH"));
    }

    /// The adjudicator's job is to catch exactly the "same base name, different
    /// role word" case; it used to carry an example that taught the opposite.
    #[test]
    fn adjudication_treats_a_shared_base_name_as_evidence_of_sameness() {
        let p = entity_adjudication_system_prompt();
        assert!(p.contains("differ only by a describing word"));
        assert!(
            !p.contains("a package and the service that consumes it"),
            "the example that produced the duplicate is back",
        );
    }
}
