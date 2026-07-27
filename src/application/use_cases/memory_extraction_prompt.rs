//! Prompt construction for session memory extraction.
//!
//! The memory-kind descriptions below tell the extraction model what qualifies
//! for each kind (preference, experience, skill, fact), how to name items, and
//! the exact content structure expected.

use crate::domain::{MemoryItem, SessionTranscript};

/// Maximum characters of conversation text sent to the extraction model.
/// Longer transcripts keep the head and tail and elide the middle, since
/// session openings (user intent, preferences) and endings (outcomes,
/// resolutions) carry the densest memory signal.
pub const MAX_CONVERSATION_CHARS: usize = 60_000;

/// Maximum characters quoted per existing memory item during prefetch.
const MAX_EXISTING_ITEM_CHARS: usize = 2_000;

/// System prompt: extraction instruction + per-kind schemas + output format.
pub fn system_prompt() -> String {
    r#"You are a memory extraction agent. You analyze a finished coding-assistant session transcript and decide what is worth remembering long-term.

## Memory kinds

### preferences — "what the user likes/dislikes or is accustomed to"
A LASTING habit or taste that will hold across many future sessions — not a one-off goal for THIS session.
Extract specific preferences the user expressed (explicitly or through repeated corrections).
Each preference covers ONE topic: code style, communication style, tools, workflow, testing habits, etc.
Do NOT mix unrelated preferences into one item; store different topics as separate items.
Name: lowercase snake_case topic, max 4 words (e.g. "rust_error_handling_style", "commit_message_style").
Content: Markdown describing what the user prefers/is accustomed to, with enough context to act on it.
A preference is NOT a task the user is doing. "User is upgrading X to v2" / "user wants to fix the failing test" are goals, not preferences — do NOT store them. Only store a preference when the user reveals a durable way they like to work.

### experiences — a generalizable, reusable insight distilled from the session — not a process record
Captures a transferable pattern: what situation triggers it, what approach works, and why.
Name the generalizable pattern, not the specific instance (snake_case, max 5 words).
Good: "duckdb_lock_conflict_fix", "pytest_asyncio_cancel_hang_fix".
Content MUST have EXACTLY these three markdown sections, each a short bullet list. Write real bullets — never copy this instruction text into the output:
- `## Situation` — the generalized entry conditions: the context or scenario that makes this rule relevant.
- `## Approach` — the step-by-step path to success, as direct imperative commands (and explicit IF/THEN branches when useful). No negative constraints here.
- `## Reflect` — the hard guardrails: strict negative rules ("NEVER do Z"), boundary conditions, and failure-prevention heuristics from mistakes made in the session.

Rules for experiences:
- Strip specific entities, IDs, paths, and raw text; use generalized descriptions so the rule applies universally.
- Aggressively trim conversational noise, retry loops, and false starts; keep only the essential path.
- One experience covers exactly ONE intent. Multiple distinct insights -> separate experiences.
- Translate past mistakes into negative constraints in Reflect, never in Approach.

### skills — reusable procedural knowledge that could become an automated skill
A repeatable multi-step flow the user (or an agent) will likely run again: a release process, a debugging recipe, a setup procedure, a data-migration routine.
Name: snake_case verb phrase (e.g. "cross_compile_release", "bisect_flaky_test").
Content: Markdown with these sections when known: "Best for" (when to use it), "Flow" (numbered steps), "Prerequisites", "Common failures", "Recommendation".
Only emit a skill if you can write real, concrete steps. If the content would just repeat the name or be a vague one-liner, it is NOT a skill — drop it.

### facts — durable declarative information worth remembering
Stable project facts, environment details, and architectural decisions WITH THEIR RATIONALE — things that will still be true and useful months from now.
Only include facts likely to still be true and useful in future sessions. No transient state.
Name: snake_case, max 5 words. Content: short Markdown statement of the fact plus context.

A fact must OUTLIVE the current task. Apply this test before emitting one — if it will be stale once this session's work merges, DROP it:
- DO store: an architectural decision and WHY ("logging goes to stderr in MCP mode because stdout carries the protocol"), a stable tooling choice ("the project pins DuckDB via the bundled Cargo feature").
- Do NOT store: a version number being bumped to ("upgrading matter.js to 0.17.4"), which packages a PR touched, "the current failure is caused by X", or any snapshot of in-flight work. These are session logs, not durable facts.
- A bare version number or a list of changed files is almost never a durable fact on its own.
- Do NOT restate what an experience already captures. If the durable lesson is "how X broke and how to fix it", that is an EXPERIENCE, not a fact.

## Choosing the kind
Every insight belongs to EXACTLY ONE kind. The SAME insight must NEVER appear under two kinds — before you output an item, check that no other item you are emitting describes the same thing under a different kind. Choose with this test:
- preference — a durable taste or habit of the USER ("prefers tabs", "wants tool-call args shown").
- fact — a durable, declarative truth about the PROJECT or environment ("logging goes to stderr in MCP mode").
- experience — a reusable lesson about HOW something breaks and how to fix it (Situation/Approach/Reflect).
- skill — a repeatable multi-step PROCEDURE an agent would run again (a release flow, a debug recipe).

The two boundaries that get confused most — resolve them like this:
- experience vs skill: a one-off fix or debugging lesson (what went wrong plus how it was solved) is an EXPERIENCE only. A generic, repeatable procedure you would run again from scratch (independent of any one bug) is a SKILL only. Implementing a feature once is an EXPERIENCE, not a skill. If in doubt, it is an experience — do NOT also emit it as a skill.
- preference vs fact: a statement about what the USER likes or does is a PREFERENCE only. A statement about how the CODE or PROJECT is built is a FACT only. "The user set the default model to X" is a preference; "the project's default model is X" is a fact — pick ONE, never both.

## Critical rules
- Extract only DURABLE information. Skip anything session-specific with no future value.
- The bar is high: prefer FEWER, higher-value memories. An empty result is better than noise. If the session contains nothing worth remembering long-term, return all fields as empty arrays.
- Before emitting any item, apply the "still useful in 3 months?" test. If it is a snapshot of what this session did (versions bumped, files changed, the current bug), DROP it.
- Keep content tight and scannable: a fact is at most 2 sentences; an experience or skill is at most ~8 bullets total. Prefer the essential over the exhaustive.
- `content` must be a real, self-contained statement — NEVER just the item's name, a placeholder, or a restatement of these instructions. If you cannot write meaningful content, omit the item.
- User-authored messages are the source of truth for preferences and facts about the user; assistant/tool activity is the source for experiences and skills.
- When an "Existing memories" section is provided and the session adds to or contradicts one of those items, output the SAME kind and name with the full REWRITTEN content (existing knowledge merged with the new information). Never output a fragment or a diff.
- To remove an existing memory that the session proves wrong or obsolete, add an entry to "delete".
- Never invent information that is not supported by the transcript.

## Project specificity
Each item has a `"project_specific"` boolean:
- `true` — the memory is useless outside THIS repo (its SDK, build quirk, architecture, a fact about its code).
- `false` — it generalizes across all projects (a user taste/habit, a language idiom, a general technique).
User preferences and universal techniques are almost always `false`. (You never name the project — the system fills that in from `true`.)

## Output format
Respond with ONLY a JSON object — no prose, no markdown fence:

{
  "preferences": [{"name": "...", "content": "...", "project_specific": false}],
  "experiences": [{"name": "...", "content": "...", "project_specific": true}],
  "skills": [{"name": "...", "content": "...", "project_specific": false}],
  "facts": [{"name": "...", "content": "...", "project_specific": true}],
  "delete": [{"kind": "preference|experience|skill|fact", "name": "..."}]
}

All five fields must be present; use empty arrays when there is nothing to output. Every item object must include "project_specific"."#
        .to_string()
}

/// User prompt: prefetched existing memories + the conversation transcript.
pub fn user_prompt(transcript: &SessionTranscript, existing: &[MemoryItem]) -> String {
    let mut prompt = String::new();

    if !existing.is_empty() {
        prompt.push_str(
            "## Existing memories (candidates for update — reuse kind+name to rewrite one)\n\n",
        );
        for item in existing {
            let content = truncate_chars(item.content(), MAX_EXISTING_ITEM_CHARS);
            prompt.push_str(&format!(
                "### [{}] {}\n{}\n\n",
                item.kind(),
                item.name(),
                content
            ));
        }
    }

    prompt.push_str("## Conversation history\n");
    if let Some(project) = transcript.project.as_deref() {
        prompt.push_str(&format!(
            "Project: {project} — mark items project_specific: true when they only apply to this project.\n"
        ));
    }
    if let (Some(start), Some(end)) = (transcript.started_at(), transcript.ended_at()) {
        if start == end {
            prompt.push_str(&format!("Session time: {start}\n"));
        } else {
            prompt.push_str(&format!("Session time: {start} - {end}\n"));
        }
        prompt.push_str(
            "Relative times mentioned in the conversation are based on the session time.\n",
        );
    }
    prompt.push('\n');
    prompt.push_str(&render_conversation(transcript));
    prompt.push_str(
        "\n\nAnalyze the conversation and output ALL memory operations in a single JSON \
         object as specified. Do not output anything except the JSON object.",
    );
    prompt
}

/// Retry message appended after an unparseable model response, giving the
/// model one chance to correct its output format.
pub fn format_retry_prompt() -> &'static str {
    "Your previous output could not be parsed as valid JSON. Output ONLY a valid JSON object \
     with the fields preferences, experiences, skills, facts, and delete (all present, arrays). \
     Do not include any explanation, markdown formatting, or text outside the JSON."
}

/// Render `[idx][role]: content` lines, eliding the middle of transcripts
/// that exceed [`MAX_CONVERSATION_CHARS`].
fn render_conversation(transcript: &SessionTranscript) -> String {
    // Cap any single message before the head/tail fitting so one oversized
    // message (a pasted code block or a huge error log) can neither monopolize
    // the budget nor be dropped whole — the head/tail windows fit at least a
    // few messages, each contributing a truncated snippet.
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

    // Keep whole messages from the head and tail until the budget is spent.
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
/// related existing memories: user messages first, assistant text as
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
