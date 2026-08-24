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

**`Memory`** — a self-contained statement plus the entities it mentions:
```
id, kind=Fact, statement, entity_ids: Vec<String>, project: Option<String>,
recorded_at, source_session_id, source_message_index, source_kind, confidence
```
Plus an embedding over `statement` in a side table. The memory↔entity link
is a `memory_entities(memory_id, entity_id)` join table.

**Update model:** hard delete + insert. No `status`, no `valid_from`/`valid_to`,
no `derived`/`derived_from`. Newest write wins. The displaced row is deleted
*after* the new one lands, so a failed write loses nothing.

**No `Predicate`.** The verb lives in the statement, where a reader looks
for it. Inspection of the live store showed most rows landed on `relates_to`
anyway — the closed vocabulary was not pulling its weight.

**No subject/object split.** A fact mentions zero or more entities via
`entity_ids`. That is the whole of the memory↔entity relationship.

**`MemoryKind`** — keep the enum with a single `Fact` variant. Keeps the field
on `Memory` for forward compatibility without carrying dead variants.

**`SourceKind`** — two variants: `user_stated`, `extracted`. The
`assistant_inferred` / `derived` split was not stable enough to keep; both
parse-forward into `extracted` for rows written by an older build.

**`Entity`** — keep. `entity_name_key` normalization stays.
Resolution = exact name-key match only. The embedding-similarity tier and the
LLM adjudication tier are deleted (the live data shows the failure mode they
were built for is better handled by the normalization we already have).

`entity_type` vocabulary: `person, tool, service, library, concept`. The
`project` type is retired — projects are a column on `Memory`, not entities.

**`ImportedSession`** — kept, with a `(source, id)` composite key. Claude,
OpenCode and Zed mint ids from independent namespaces, so `id` alone is not
unique.

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
- `memory_resources` — (uri, source, name, abstract, overview, content,
  created_at) with the embedding (over `abstract + overview`) in
  `memory_resource_embeddings(uri, vector)`.
- `memory_sessions` — unchanged from current.
- `memory_meta` — schema version.

### Extraction

Prompt asks the model for self-contained facts plus entity mentions:
`(statement, source_kind, confidence, source_message_index, entity_mentions:
[{name, type}])`.

No `kind`, no `predicate`, no `subject`/`object` split. Server resolves each
mention to an entity via `entity_name_key`, creating it on first sight.
Entity types outside `VALID_ENTITY_TYPES` map to `unknown`.

### Recall

Single ranking. RRF over:
- cosine rank on `statement` embedding
- recency rank on `recorded_at`

Optional `project` filter. No kind filter (only one kind). No tree traversal.

### MCP tools

Tool names stay — external clients depend on them. Internals are reimplemented
over the new model and some parameters become no-ops:

- `search_memories(query, limit, kind?, project?, namespace?)` — `kind` accepted
  but ignored (only `Fact` exists). Returns flat facts ranked by
  RRF(cosine, recency). No more `provenance` block (no edges).
- `list_memories(kind?, status?)` — both parameters accepted, both ignored.
  Returns all memories newest first.
- `resume_work(project?, namespace?, limit?)` — unchanged.
- `read_memory(id)` — returns the memory, no `edges` array. `memory://` URI
  form is accepted for resources only (returns the resource row); sessions and
  the old memory digest return not_found.
- `browse_memory(uri?)` — kept as a thin listing over the three remaining
  roots: `memory://memory` (fact count), `memory://sessions` (recent
  sessions), `memory://resources` (stored resources). No L0/L1/L2 abstracts.
- `forget_memory(id)` — hard delete now. Returns `{ "deleted": true }`.
- `add_resource(source, name?)` — kept, backed by a new minimal
  `memory_resources(uri, source, name, abstract, overview, content, created_at)`
  table. The LLM still writes both the one-line `abstract` (L0, used for
  display + embedding) and the longer `overview` (L1, an orientation read
  before deciding whether to open `content`). What goes away is the
  `memory://` *tree* and the `MemoryNode` type, not the summaries.
- `list_namespaces`, `create_namespace`, `assign_project` — unchanged.

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
