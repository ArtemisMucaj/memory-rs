---
name: memory-rs-mcp
description: Use before implementing a feature, refactoring, fixing a bug, continuing work already under way, or answering a question about this project — and whenever the user refers back to earlier work, a decision already taken, or how they like things done. Load it first to recover what previous sessions established (the goal, the decisions, what broke and how it was fixed) instead of asking them to re-explain. Remembers across sessions what a fresh context has lost.
metadata:
  author: ArtemisMucaj
  version: "1.0.0"
compatibility: Requires the memory-rs MCP server to be connected (`memory-rs mcp` over stdio, or `memory-rs serve` which mounts MCP at `/mcp`) and at least one imported session.
---

# Memory (MCP)

The memory-rs MCP server exposes a long-term memory store as tools you can call
directly. This skill is the playbook for *when and in what order* to call them.
It names each tool and what it is for; the exact parameters are on each tool's
own schema — discover those as you go, don't guess them from here.

Three capabilities over one store:

- Catch up — pick up where the last session stopped (`resume_work`).
- Recall — hybrid retrieval over atomic memories, scoped to a project or a
  namespace of related repositories (`search_memories`, `list_memories`,
  `read_memory`, `browse_memory`).
- Record — keep the store honest and useful (`forget_memory`, `add_resource`,
  `list_namespaces` / `create_namespace` / `assign_project`).

Catch up *before* you recall. `resume_work` answers "where was I" without you
knowing what to ask for, which is exactly what a fresh context has lost;
`search_memories` answers "what do I know about X" and needs you to already have
X.

---

# The runbook

Follow these phases in order. Scope is the thing to get right: a project argument
means one repository, a namespace means a group of them (a multi-repo effort),
and omitting both spans everything. Memories with no project are global and are
always included. Pass one or the other, never both.

## Phase 1 — Catch up on recent work

Do this before touching code in a repository you have not worked in this session.

- `resume_work` — the recent sessions in scope: when each ran, a one-line
  summary, the arc of what happened, and the durable memories it produced. Read
  it and continue from there instead of asking the user to restate the goal.
  Scope it to the project you are about to work in.
- `list_memories` with the preference kind — the user's standing habits and
  tastes, loaded once so you work the way they expect.

`resume_work` makes no model call: the summaries were written when each session
was imported, so there is no cost reason to skip it.

## Phase 2 — Recall what bears on the task

Call `search_memories` with the *subject* you need, in context — not the words
you expect the text to contain.

- Good: "how the event pipeline handles malformed payloads", "why logging goes to
  stderr".
- Weak: "error" (too generic — say "error handling for X").

Narrow when results are noisy (restrict the kind: `preference` for the user's
tastes, `fact` for how the code is built, `experience` for a lesson about
something breaking, `skill` for a procedure), widen the limit when they are thin,
and switch project → namespace when the answer may live in a sibling repository.
Those are all parameters on the tool — check its schema for the names.

Only current memories come back: superseded and retracted ones never surface, so
a hit is something the store still believes.

## Phase 3 — Read the detail behind a hit

- `read_memory` — one memory with its typed edges, so *why* it is believed comes
  back in the same call: what it superseded, what refines it, what contradicts
  it. Also reads a `memory://` node (a session's summary, then its transcript).
- `browse_memory` — the store as a small filesystem: the roots, or a directory's
  children with their one-line abstracts. Use it to find a session or resource
  when you don't have an id.

A memory carrying a `contradicts` edge is not settled. Report the disagreement
rather than asserting one side.

## Phase 4 — Keep the store honest

- `forget_memory` — retract something this session proved wrong. It is a
  retraction, not a deletion: the memory stays for provenance and stops being
  recalled. Say "retracted", not "deleted".
- `add_resource` — store a file or URL (a spec, a runbook) as recallable
  material alongside the memories.
- `list_namespaces` / `create_namespace` / `assign_project` — group repositories
  that are worked on together, so one recall spans the whole effort.

Importing sessions is not exposed as a tool: it is a background job the server
runs (its dream cycle harvests finished sessions), or a deliberate CLI/API call.
If the store looks like it is missing this session's work, that is expected until
the session ends and is harvested.

---

# Reference

## Tool index (by phase)

| Phase | Tools |
|---|---|
| Catch up | `resume_work`, `list_memories` (preference kind) |
| Recall | `search_memories`, `list_memories`, `read_memory`, `browse_memory` |
| Record | `forget_memory`, `add_resource`, `list_namespaces` / `create_namespace` / `assign_project` |

Parameters for each tool live on its schema — discover them at call time rather
than assuming. Prefer omitting optional filters (kind, limit, scope) unless they
are needed.

## Composing tools

Most questions are answered by combining a few calls rather than one.

- Start a session in a repo — `resume_work` scoped to the project, then
  `list_memories` for preferences. That is the whole warm-up; go straight to the
  task afterwards.
- Understand a decision — `search_memories` for the subject, then `read_memory`
  on the best hit to see what it superseded and what contradicts it. The edges
  are the difference between "this is true" and "this was decided, and here is
  what it replaced".
- Reconstruct a session — `resume_work` gives the summary; `read_memory` on
  `memory://sessions/<id>` gives the transcript when you need the detail behind
  it.
- Span several repositories — assign the projects to a namespace once
  (`assign_project`), then pass that namespace to `resume_work` and
  `search_memories` instead of repeating per-project calls.

## Getting good results

- Start every task at Phase 1: catch up before recalling, and recall before
  asking the user to explain.
- Scope to the project you are working in; widen only when a memory could be
  global or live in a sibling repository. Sessions imported before projects were
  tracked carry none and appear unscoped only.
- Treat a hit as a claim, not a fact: check its kind (a `preference` is the
  user's taste, a `fact` is about the code) and look for a `contradicts` edge
  before acting on it.
- Discover each tool's parameters from its schema at call time; don't assume
  argument names or invent filters that may not exist.

## Keywords

mcp, model context protocol, memory mcp, long-term memory, resume work, catch
up, what was I doing, pick up where I left off, previous session, session
history, session summary, recall, remember, user preferences, coding style,
durable facts, project decisions, lessons learned, experiences, skills, semantic
search, hybrid search, memory graph, append-only, supersede, retract,
contradiction, provenance, entities, namespace, project scope, resource, dream,
consolidation, memory://, resume_work, search_memories, list_memories,
read_memory, browse_memory, forget_memory, add_resource
