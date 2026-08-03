---
name: memory-rs-cli
description: Use before implementing a feature, refactoring, fixing a bug, continuing work already under way, or answering a question about this project — and whenever the user refers back to earlier work, a decision already taken, or how they like things done. Load it first to recover what previous sessions established (the goal, the decisions, what broke and how it was fixed) instead of asking them to re-explain. Remembers across sessions what a fresh context has lost.
metadata:
  author: ArtemisMucaj
  version: "1.0.0"
compatibility: Requires the memory-rs binary and at least one imported session (`memory-rs import`). LLM-driven commands (`import`, `dream`, `add`) need an OpenAI-compatible endpoint in `config.json` or the `OPENAI_*` environment.
---

# Memory

A CLI that gives an AI assistant three capabilities over one long-term store:

- Catch up — pick up where the last session stopped: `resume` returns the recent
  sessions in a project, what each was about, and the durable memories they left
  behind.
- Recall — hybrid retrieval (embeddings + keyword) over atomic memories, scoped
  to a project or a namespace of related repositories: `search`, `list`, `show`.
- Record — turn a finished session into durable memory: `import`, then `dream` to
  consolidate; `conflicts` and `delete` keep the store honest.

Catch up *before* you recall. `resume` answers "where was I" without you knowing
what to ask for, which is exactly what a fresh context has lost; `search` answers
"what do I know about X" and needs you to already have X.

---

# The runbook

Follow these phases in order. Scope flags are the thing to get right: `--project`
is one repository, `--namespace` is a group of them (a multi-repo effort), and
omitting both spans everything. Memories with no project are global and always
included.

## Phase 1 — Catch up on recent work

Do this before touching code in a repository you have not worked in this session.

```shell
memory-rs resume --project owner/repo        # recent sessions + what they left behind
memory-rs resume --namespace platform        # across a multi-repo effort
memory-rs resume --limit 10                  # widen the window (default 5)
memory-rs resume --project owner/repo -F json
```

Each entry is one session: when it ran, a one-line summary, the arc of what
happened, and the memories it produced. Read it and continue from there rather
than asking the user to restate the goal.

Then load the standing preferences once, so you work the way the user expects:

```shell
memory-rs list --kind preference             # durable habits and tastes
```

## Phase 2 — Recall what bears on the task

Once you know what you are doing, ask for the parts that matter. Describe the
*subject*, not the words you expect to find.

```shell
# Good — a subject with context
memory-rs search "how the event pipeline handles malformed payloads" --project owner/repo
memory-rs search "why logging goes to stderr"

# Weak — fix by adding the subject
memory-rs search "error"                     # too generic → "error handling for X"
```

Narrow when the result set is noisy, widen when it is thin:

```shell
memory-rs search "..." --kind experience     # only reusable lessons
memory-rs search "..." --kind fact           # only durable project facts
memory-rs search "..." --num 25              # default 10
memory-rs search "..." --namespace platform  # the whole multi-repo effort
memory-rs search "..." -F json               # structured output when you'll parse it
```

Only current memories come back — superseded and retracted ones never surface,
so a hit is something the store still believes.

## Phase 3 — Read the detail behind a hit

```shell
memory-rs show <memory-id>                   # one memory with its typed edges
memory-rs show memory://sessions/<id>        # a session: summary, then transcript
memory-rs tree                               # the store's roots
memory-rs tree memory://sessions             # sessions with their one-line abstracts
```

`show` on a memory is how you see *why* it is believed: what it superseded, what
refines it, what contradicts it. A memory with a `contradicts` edge is not
settled — say so rather than asserting it.

## Phase 4 — Record what this session established

Memory only helps the next session if this one is written down.

```shell
memory-rs import /path/to/transcript.jsonl   # extract durable memories from a session
memory-rs import /path/to/transcript.jsonl --force   # re-import (clears its prior memories)
memory-rs dream                              # harvest finished sessions + consolidate
memory-rs conflicts                          # unresolved disagreements, both still active
memory-rs delete <memory-id>                 # retract something the session proved wrong
```

`import` and `dream` call the LLM. `delete` retracts rather than erases: the
store is append-only, so a retracted memory stays for provenance and simply
stops being recalled.

---

# Reference

## First-time setup

memory-rs has no published release assets yet, so build it from a checkout:

```shell
cargo build --release        # binary at target/release/memory-rs
memory-rs --version
```

Point it at an OpenAI-compatible endpoint (LM Studio, vLLM, hosted OpenAI) for
the LLM-driven commands — a named endpoint in `~/.memory-rs/config.json`, or the
`OPENAI_BASE_URL` / `OPENAI_MODEL` / `OPENAI_API_KEY` environment as the
fallback. Chat and embeddings resolve independently, so a hosted chat model can
pair with a local embedder.

## Namespaces — grouping repositories

A namespace is a set of projects recalled together, for an effort that spans
several repositories.

```shell
memory-rs namespace create platform
memory-rs namespace assign platform owner/repo
memory-rs namespace list
memory-rs namespace show platform
```

## Store inspection

```shell
memory-rs sessions                           # what has been imported (and what failed)
memory-rs stats                              # counts by kind and status
memory-rs add ./NOTES.md                     # store a file or URL as a recallable resource
```

## Long-running modes

```shell
memory-rs serve --port 8766                  # REST API + MCP over HTTP (one port)
memory-rs mcp                                # MCP over stdio, for an assistant
memory-rs tui                                # interactive browser + import
```

Rarely needed global flags: `--data-dir <path>` (default `~/.memory-rs`),
`--openai-endpoint <name>` (use a named endpoint instead of the active one).

## Getting good results

- Start at Phase 1. `resume` is cheap — no LLM call, it reads summaries written
  at import — so there is no reason to skip it.
- Prefer scoping to the project you are in; drop to unscoped only when a memory
  might be global. Sessions imported before projects were tracked carry none and
  appear unscoped only.
- Treat a hit as a claim, not a fact: check the kind (`preference` is the user's
  taste, `fact` is about the code) and look for a `contradicts` edge before
  acting on it.
- Write the session down. A session that is never imported teaches the store
  nothing, and the next context starts from zero again.

## Keywords

long-term memory, memory store, resume work, catch up, what was I doing, pick up
where I left off, previous session, session history, session summary, recall,
remember, user preferences, coding style, durable facts, project decisions,
architecture decision, lessons learned, experiences, skills, semantic search,
hybrid search, embeddings, memory graph, append-only, supersede, retract,
contradiction, conflicts, provenance, entities, namespace, project scope,
import session, transcript, dream, consolidation, digest, memory://, virtual
filesystem, L0 abstract, L1 overview, MCP, TUI
