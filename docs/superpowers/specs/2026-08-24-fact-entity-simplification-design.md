# memory-rs simplification — fact + entity only

**Issue:** [ArtemisMucaj/memory-rs#24](https://github.com/ArtemisMucaj/memory-rs/issues/24)

**Goal:** Strip memory-rs down to the two things that carry value — facts about
entities — and drop everything that adds code, prompt tokens and schema
surface without paying for itself.

## Motivation

Inspection of the live store (`~/.memory-rs/memory.duckdb`) shows:

- 4 of 5 entities are `entity_type = 'project'`, duplicating the projects
  view. Projects are already carried by `Memory.project`; reifying them as
  entities buys nothing.
- `memory_edges` is empty in practice. The typed edge vocabulary
  (`supersedes`, `contradicts`, `refines`, `retracts`, `corroborates`,
  `relates_to`) is dead schema.
- `memory_items`, `memory_nodes`, `memory_node_vectors`, `memory_namespaces`,
  `memory_dream_runs` exist only to feed the L0/L1/L2 `memory://` tree, which
  nothing consumes outside the TUI.
- `MemoryKind` has 4 variants, of which `Skill` and `Experience` are
  unreliable to extract (per the issue). Only `Fact` is consistently useful.
- `Predicate` has 17 variants; the live data fits comfortably in 7.
- `MemoryStatus` + `valid_from`/`valid_to` exist only to support edge-based
  supersession. With edges gone, hard delete is simpler and loses nothing.

## Design decisions

### Data model that survives

**`Memory`** — append-only-ish fact triple:
```
id, kind=Fact, subject: EntityRef, predicate: Predicate, object: EntityRef,
statement, project: Option<String>, recorded_at, source_session_id,
source_message_index, source_kind, confidence
```
Plus an embedding over `statement` in a side table.

**Update model:** hard delete + insert. No `status`, no `valid_from`/`valid_to`,
no `derived`/`derived_from`. Newest write wins.

**`Predicate`** — 17 variants → 7:
`prefers, avoids, uses, fixes, decided, is_a, relates_to`.
Dropped: `requires, provides, implements, contains, derived_from, configures,
causes, prevents, has, works_on`. Callers mapping old values: `requires /
provides / implements / contains / derived_from / configures / has / works_on`
fold into `relates_to`; `causes / prevents` fold into `fixes` when the object
is a problem, else `relates_to`.

**`MemoryKind`** — keep the enum with a single `Fact` variant. Keeps the field
on `Memory` for forward compatibility without carrying dead variants.

**`Entity`** — keep. `entity_name_key` normalization stays.
Resolution = exact name-key match only. The embedding-similarity tier and the
LLM adjudication tier are deleted (the live data shows the failure mode they
were built for is better handled by the normalization we already have).

`entity_type` vocabulary: `person, tool, service, library, concept`. The
`project` type is retired — projects are a column on `Memory`, not entities.

**`ImportedSession`** — keep as-is. Sessions still drive "what was I working
on" and feed extraction.

### Data model deleted

- `MemoryItem`, `MemoryOperation` — the parallel item store.
- `MemoryNode`, `NodeKind` — the L0/L1/L2 `memory://` tree.
- `MemoryEdge`, `EdgeType`, `EdgeOrigin` — the edge vocabulary.
- `MemoryStatus`, `DreamRun`, `SessionStatus`, `MemoryStoreStats` — status /
  stats types that existed to feed deleted surfaces.
- Tables: `memory_items`, `memory_nodes`, `memory_node_vectors`,
  `memory_edges`, `memory_namespaces`, `memory_dream_runs`.

### Schema

Drop and recreate `memory.duckdb`. **No migration.** New tables:

- `memories` — one row per fact (schema in plan).
- `entities` — id, entity_type, canonical_name, created_at, updated_at.
- `entity_names` — (entity_id, name, name_key) for alias resolution.
- `memory_embeddings` — (memory_id, embedding VECTOR).
- `memory_sessions` — unchanged from current.
- `memory_meta` — schema version.

### Extraction

Prompt asks the model for triples:
`(subject_name, subject_type, predicate, object_name_or_literal, statement,
project, source_message_index)`.

Predicate vocabulary restricted to the 7 above. No kinds, no L0/L1/L2
abstracts. Server resolves `subject_name` / `object_name` to entities via
`entity_name_key`, creating entities on first sight.

### Recall

Single ranking. RRF over:
- cosine rank on `statement` embedding
- recency rank on `recorded_at`

Optional `project` filter. No kind filter (only one kind). No tree traversal.

### MCP tools

Three tools:
- `memory_recall(query, project?, limit?)` — flat list of facts.
- `memory_entity(name)` — entity drill-down: canonical entity + all facts
  where it appears as subject or object.
- `memory_sessions(limit?)` — recent sessions.
- `memory_resume(project?)` — existing resume briefing, kept.

### Dream cycle

Keep the harvest phase (import new finished sessions). Dedup happens at
write-time via the natural key (subject_key, predicate, object_key). The
consolidation pass is deleted — no edges to write, no contradiction detection,
no clustering.

### Surfaces

- **CLI** — keep `import`, `recall`, `entity`, `sessions`, `resume`, `dream`
  (harvest only). Drop subcommands tied to items/nodes/edges.
- **TUI** — flat fact list + entity drill only. Drop kind tabs, edge views,
  memory-tree views.
- **HTTP API** — drop endpoints for kinds/edges/nodes/tree. Keep
  `/api/memory/recall`, `/api/memory/entity/:name`, `/api/memory/sessions`,
  `/api/memory/resume`, `/api/memory/import`.
- **MCP** — the 4 tools above.

## Out of scope

- Migration of existing databases.
- Embedding backend changes (stays on `openai_rs`).
- Session transcript parsing (unchanged).
- New entity-resolution tiers (deliberately reduced to name-key only).

## Success criteria

- `cargo build` and `cargo test` pass.
- Live DB can be deleted and rebuilt from scratch via `memory import` over
  existing session transcripts.
- `memory_recall` returns facts ranked by RRF(cosine, recency).
- Extraction prompt fits in roughly half the tokens of the current one
  (measured by counting prompt template lines).
