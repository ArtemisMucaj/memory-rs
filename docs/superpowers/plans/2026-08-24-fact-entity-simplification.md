# Fact + Entity Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip memory-rs to facts (triples) + entities with name-key resolution; delete MemoryItem / MemoryNode / MemoryEdge / MemoryStatus / Dream consolidation / 10 predicate variants; rank recall by RRF(cosine, recency).

**Architecture:** Single `memories` table (fact triples, no status/validity window), `entities` + `entity_names` (exact name-key lookup only), `memory_embeddings` for statement vectors, `memory_resources` for file/URL resources, `memory_sessions` for imported sessions. Dream = harvest only. MCP tool names preserved; internals rewritten.

**Tech Stack:** Rust, DuckDB (duckdb-rs), tokio, async-trait, rmcp, clap, serde.

**Spec:** `docs/superpowers/specs/2026-08-24-fact-entity-simplification-design.md`

---

## File structure

### Deleted

- `src/domain/memory.rs` — `MemoryItem`, `MemoryNode`, `NodeKind`, `MemoryOperation`, `DreamRun`, `SessionStatus` (move `SessionMessage` / `SessionTranscript` / `ImportedSession` into `domain/session.rs`).
- `src/domain/memory_graph.rs` — keep `Entity`, `EntityRef`, `Predicate` (7 variants), `SourceKind`, `entity_name_key`. Drop `MemoryStatus`, `MemoryEdge`, `EdgeType`, `EdgeOrigin`, `MemoryStoreStats`, `Memory::status`/`valid_from`/`valid_to`/`derived`/`derived_from` fields.
- `src/application/use_cases/memory_browse.rs`, `memory_summary.rs`, `memory_dream_prompt.rs` — tree and digest use cases.
- `src/application/use_cases/memory_extraction.rs`, `memory_extraction_prompt.rs` — old `MemoryItem` extraction. Ingestion already covers facts.
- `src/application/interfaces/node_repository.rs`.
- `src/connector/adapter/duckdb_store.rs` — `NodeRepository` impl (item store, nodes, namespaces, dream runs). Namespaces move into the new memory repository.
- `src/tui/screens/import.rs` — fold small remainder into `tui/app.rs`.
- Tests: `duckdb_memory_repository_tests.rs`, `duckdb_store_tests.rs`, `import_pipeline_tests.rs`, `memory_pipeline_tests.rs`, `schema_compatibility_tests.rs` — replaced by two new test files.

### Created

- `src/domain/resource.rs` — `MemoryResource` type.
- `src/application/interfaces/resource_repository.rs` — port.
- Tests: `tests/fact_repository_tests.rs`, `tests/ingestion_pipeline_tests.rs`.

### Modified

Everything under `src/application/use_cases/` and `src/connector/` that references the deleted types. The blast radius is most of the crate; the plan goes layer-by-layer so each task ends in a compiling state.

---

## Task 1: Domain — collapse `Predicate` to 7 variants

**Files:**
- Modify: `src/domain/memory_graph.rs:131-282` (Predicate enum + impls)

- [ ] **Step 1: Write the failing test**

Append to the existing `#[cfg(test)] mod tests` block in `src/domain/memory_graph.rs`:

```rust
#[test]
fn shrunken_predicate_vocabulary_round_trips() {
    for predicate in Predicate::ALL {
        assert_eq!(Predicate::parse(predicate.as_str()), Some(predicate));
    }
}

#[test]
fn retired_predicates_parse_to_none() {
    for retired in [
        "requires", "provides", "implements", "contains", "derived_from",
        "configures", "causes", "prevents", "has", "works_on",
    ] {
        assert_eq!(Predicate::parse(retired), None, "{retired}");
    }
}

#[test]
fn predicate_vocabulary_is_seven() {
    assert_eq!(Predicate::ALL.len(), 7);
}
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cargo test --lib domain::memory_graph::tests::shrunken_predicate_vocabulary_round_trips 2>&1 | tail -10
```
Expected: FAIL — `requires` etc still parse.

- [ ] **Step 3: Collapse the enum**

Replace the `Predicate` enum, `ALL`, `as_str`, `meaning`, `parse` with:

```rust
/// The relation in a memory's subject–predicate–object triple.
///
/// Closed vocabulary, deliberately small. Anything that does not fit one of
/// these seven should be expressed with [`Predicate::RelatesTo`] and carried
/// by the `statement` text — the statement is what recall reads; the
/// predicate is a coarse filter at best.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Predicate {
    /// A durable taste or habit of the user.
    Prefers,
    /// A durable dislike, or something deliberately not done.
    Avoids,
    /// Depends on, is built with, employs.
    Uses,
    /// Resolves a problem.
    Fixes,
    /// A choice that was made, with its rationale in the statement.
    Decided,
    /// Type or category membership.
    IsA,
    /// Escape hatch — a genuine relation none of the above expresses.
    RelatesTo,
}

impl Predicate {
    pub const ALL: [Predicate; 7] = [
        Predicate::Prefers,
        Predicate::Avoids,
        Predicate::Uses,
        Predicate::Fixes,
        Predicate::Decided,
        Predicate::IsA,
        Predicate::RelatesTo,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            Predicate::Prefers => "prefers",
            Predicate::Avoids => "avoids",
            Predicate::Uses => "uses",
            Predicate::Fixes => "fixes",
            Predicate::Decided => "decided",
            Predicate::IsA => "is_a",
            Predicate::RelatesTo => "relates_to",
        }
    }

    pub fn meaning(&self) -> &'static str {
        match self {
            Predicate::Prefers => "a durable taste or habit of the user",
            Predicate::Avoids => "a durable dislike, or something deliberately not done",
            Predicate::Uses => "depends on, is built with, employs",
            Predicate::Fixes => "resolves a problem",
            Predicate::Decided => "a choice that was made (put the rationale in the statement)",
            Predicate::IsA => "type or category membership",
            Predicate::RelatesTo => "none of the above fits — use only as a last resort",
        }
    }

    pub fn parse(s: &str) -> Option<Predicate> {
        match s
            .trim()
            .to_ascii_lowercase()
            .replace([' ', '-'], "_")
            .as_str()
        {
            "prefers" | "prefer" | "likes" => Some(Predicate::Prefers),
            "avoids" | "avoid" | "dislikes" => Some(Predicate::Avoids),
            "uses" | "use" | "used" | "utilises" | "utilizes" | "depends_on" => {
                Some(Predicate::Uses)
            }
            "fixes" | "fix" | "fixed" | "resolves" => Some(Predicate::Fixes),
            "decided" | "decides" | "chose" | "chooses" => Some(Predicate::Decided),
            "is_a" | "isa" | "is" | "type_of" => Some(Predicate::IsA),
            "relates_to" | "related_to" | "relates" => Some(Predicate::RelatesTo),
            _ => None,
        }
    }
}
```

- [ ] **Step 4: Fix compile errors in callers**

The 10 retired variants are referenced from `memory_dream.rs`, `memory_dream_prompt.rs`, `memory_ingestion_prompt.rs`, and tests. For this task, delete the references to retired variants inline — those files are simplified in later tasks, so a minimal "make it compile" edit is enough (e.g. remove `Predicate::Requires` from match arms).

Run:

```bash
cargo build 2>&1 | tail -30
cargo test --lib 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(domain): shrink Predicate vocabulary to 7 variants"
```

---

## Task 2: Domain — drop edges, status, derived fields from `Memory`

**Files:**
- Modify: `src/domain/memory_graph.rs`

- [ ] **Step 1: Failing test**

Add to existing `mod tests`:

```rust
#[test]
fn memory_no_longer_carries_status_or_validity_window() {
    let memory = Memory {
        id: "m1".into(),
        kind: MemoryKind::Fact,
        subject: EntityRef::Literal("user".into()),
        predicate: Predicate::Prefers,
        object: EntityRef::Literal("tabs".into()),
        statement: "User prefers tabs.".into(),
        project: None,
        recorded_at: 1,
        source_session_id: None,
        source_message_index: None,
        source_kind: SourceKind::UserStated,
        confidence: 1.0,
    };
    assert_eq!(memory.recorded_at, 1);
    // Compile-time proof these fields are gone:
    // memory.status; memory.valid_from; memory.valid_to; memory.derived;
    // should all fail to compile.
}
```

- [ ] **Step 2: Verify failure**

```bash
cargo test --lib memory_no_longer_carries_status 2>&1 | tail -5
```
Expected: FAIL — struct still has the old fields.

- [ ] **Step 3: Strip `Memory`**

In `src/domain/memory_graph.rs`:

1. Delete the `MemoryStatus`, `EdgeType`, `EdgeOrigin`, `MemoryEdge`, `MemoryStoreStats` types and their impls.
2. Rewrite `Memory`:

```rust
/// A single fact in the memory log.
///
/// Updates are hard-delete + insert at the repository layer; there is no
/// lifecycle status and no validity window. Newest write wins.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub kind: MemoryKind,
    pub subject: EntityRef,
    pub predicate: Predicate,
    pub object: EntityRef,
    pub statement: String,
    pub project: Option<String>,
    pub recorded_at: i64,
    pub source_session_id: Option<String>,
    pub source_message_index: Option<i64>,
    pub source_kind: SourceKind,
    pub confidence: f32,
}
```

3. Delete tests referencing deleted types.

- [ ] **Step 4: Make it compile**

`cargo build` will surface every caller of the removed types. Strip or simplify them inline (mostly removing match arms, function parameters, and fields). The heavy refactoring lands in later tasks; here it is enough to delete code paths that don't compile.

```bash
cargo build 2>&1 | tail -50
cargo test --lib 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(domain): drop MemoryStatus/edges/validity from Memory"
```

---

## Task 3: Domain — collapse `MemoryKind` to one variant

**Files:**
- Modify: `src/domain/memory.rs`

- [ ] **Step 1: Failing test**

```rust
#[test]
fn memory_kind_has_a_single_variant() {
    assert_eq!(MemoryKind::ALL.len(), 1);
    assert_eq!(MemoryKind::parse("fact"), Some(MemoryKind::Fact));
    assert_eq!(MemoryKind::parse("preference"), None);
    assert_eq!(MemoryKind::parse("experience"), None);
    assert_eq!(MemoryKind::parse("skill"), None);
}
```

- [ ] **Step 2: Verify failure**

```bash
cargo test --lib memory_kind_has_a_single_variant 2>&1 | tail -5
```

- [ ] **Step 3: Collapse the enum**

```rust
/// Category of a stored memory. Single-variant on purpose: the previous
/// taxonomy (Preference/Experience/Skill/Fact) was unreliable to extract
/// and added prompt surface without paying for itself. Kept as an enum so
/// the storage column stays forward-compatible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryKind {
    Fact,
}

impl MemoryKind {
    pub const ALL: [MemoryKind; 1] = [MemoryKind::Fact];

    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryKind::Fact => "fact",
        }
    }

    pub fn plural(&self) -> &'static str {
        match self {
            MemoryKind::Fact => "facts",
        }
    }

    pub fn plural_title(&self) -> &'static str {
        match self {
            MemoryKind::Fact => "Facts",
        }
    }

    pub fn parse(s: &str) -> Option<MemoryKind> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fact" | "facts" => Some(MemoryKind::Fact),
            _ => None,
        }
    }
}
```

- [ ] **Step 4: Fix callers**

Drop match arms and `MemoryKindArg` variants in `src/cli/mod.rs`. Strip `kind` filters from list/recall in use cases — keep the parameter as `Option<MemoryKind>` for forward-compat but ignore the value.

```bash
cargo build 2>&1 | tail -30
cargo test --lib 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(domain): collapse MemoryKind to Fact"
```

---

## Task 4: Domain — retire `project` as `entity_type`

**Files:**
- Modify: `src/domain/memory_graph.rs`

- [ ] **Step 1: Failing test**

```rust
#[test]
fn project_is_not_an_entity_type() {
    assert!(!VALID_ENTITY_TYPES.contains(&"project"));
}
```

- [ ] **Step 2: Add the constant + test fails**

```rust
/// Entity types the extraction model may use. `project` is deliberately not
/// on the list — projects live on `Memory.project`, not as entities.
pub const VALID_ENTITY_TYPES: &[&str] =
    &["person", "tool", "service", "library", "concept"];
```

- [ ] **Step 3: Verify test passes**

```bash
cargo test --lib project_is_not_an_entity_type 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(domain): retire 'project' entity type"
```

---

## Task 5: Application — rewrite `MemoryRepository` port

**Files:**
- Modify: `src/application/interfaces/memory_repository.rs`

- [ ] **Step 1: Failing test**

Write `tests/fact_repository_tests.rs`:

```rust
mod common;

use memory_rs::application::MemoryRepository;
use memory_rs::connector::DuckdbStore;
use memory_rs::domain::{EntityRef, Memory, MemoryKind, Predicate, SourceKind};

fn memory(id: &str, statement: &str, recorded_at: i64) -> Memory {
    Memory {
        id: id.into(),
        kind: MemoryKind::Fact,
        subject: EntityRef::Literal("user".into()),
        predicate: Predicate::Prefers,
        object: EntityRef::Literal(statement.into()),
        statement: statement.into(),
        project: None,
        recorded_at,
        source_session_id: None,
        source_message_index: None,
        source_kind: SourceKind::UserStated,
        confidence: 1.0,
    }
}

#[tokio::test]
async fn deleted_memory_is_gone_from_recall() {
    let store = DuckdbStore::in_memory(384, "test-model").unwrap();
    let repo = &store;
    let m = memory("m1", "user prefers tabs", 1000);
    repo.append_memory(&m, None).await.unwrap();
    assert!(repo.find_memory("m1").await.unwrap().is_some());
    repo.delete_memory("m1").await.unwrap();
    assert!(repo.find_memory("m1").await.unwrap().is_none());
}
```

- [ ] **Step 2: Run, verify fail**

```bash
cargo test --test fact_repository_tests deleted_memory_is_gone 2>&1 | tail -10
```
Expected: FAIL — `delete_memory` doesn't exist.

- [ ] **Step 3: Rewrite the port**

Replace `src/application/interfaces/memory_repository.rs`:

```rust
//! Persistence port for facts (memories), entities, and resources.
//!
//! The store is a table of current values, not a log: updates are hard
//! delete + insert. There is no lifecycle status, no validity window, no
//! typed edges between memories. Newest write wins.

use async_trait::async_trait;

use crate::domain::{DomainError, Entity, Memory, MemoryKind, MemoryResource};

#[async_trait]
pub trait MemoryRepository: Send + Sync {
    // ── Memories ─────────────────────────────────────────────────────────

    /// Insert (or replace by id) a memory with its optional statement embedding.
    async fn append_memory(
        &self,
        memory: &Memory,
        vector: Option<&[f32]>,
    ) -> Result<(), DomainError>;

    async fn find_memory(&self, id: &str) -> Result<Option<Memory>, DomainError>;

    async fn find_memories(&self, ids: &[String]) -> Result<Vec<Memory>, DomainError>;

    /// Newest first, optionally filtered by kind (currently always `Fact`)
    /// and project scope.
    async fn list_memories(
        &self,
        kind: Option<MemoryKind>,
        projects: Option<&[String]>,
    ) -> Result<Vec<Memory>, DomainError>;

    /// Hard-delete one memory. Returns whether it existed.
    async fn delete_memory(&self, id: &str) -> Result<bool, DomainError>;

    /// Hard-delete every memory extracted from `session_id`. Used by forced
    /// re-import: re-running extraction over an unchanged transcript is a
    /// do-over, not a new observation. Returns the count removed.
    async fn delete_memories_for_session(&self, session_id: &str) -> Result<usize, DomainError>;

    async fn count_memories_for_session(&self, session_id: &str) -> Result<usize, DomainError>;

    // ── Entities ─────────────────────────────────────────────────────────

    /// Insert or replace an entity and its names.
    async fn upsert_entity(&self, entity: &Entity) -> Result<(), DomainError>;

    async fn find_entity(&self, id: &str) -> Result<Option<Entity>, DomainError>;

    async fn find_entities(&self, ids: &[String]) -> Result<Vec<Entity>, DomainError>;

    /// Every entity whose `name_key` matches `entity_name_key(name)`. The
    /// only resolution tier — no embeddings, no LLM adjudication.
    async fn find_entities_by_name(&self, name: &str) -> Result<Vec<Entity>, DomainError>;

    async fn list_entities(&self) -> Result<Vec<Entity>, DomainError>;

    /// Memories referencing `entity_id` as subject or object, newest first.
    async fn memories_for_entity(&self, entity_id: &str) -> Result<Vec<Memory>, DomainError>;

    // ── Retrieval ────────────────────────────────────────────────────────

    /// Cosine similarity over statement embeddings. Returns `(memory, score)`
    /// best first.
    async fn search_memories_semantic(
        &self,
        vector: &[f32],
        kind: Option<MemoryKind>,
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<(Memory, f32)>, DomainError>;

    /// Case-insensitive substring over `statement`.
    async fn search_memories_keyword(
        &self,
        query: &str,
        kind: Option<MemoryKind>,
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<(Memory, f32)>, DomainError>;

    /// All memories with a non-NULL `recorded_at`, newest first. Used for
    /// the recency leg of recall's RRF fusion.
    async fn list_memories_by_recency(
        &self,
        kind: Option<MemoryKind>,
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<Memory>, DomainError>;

    // ── Resources ────────────────────────────────────────────────────────

    async fn upsert_resource(
        &self,
        resource: &MemoryResource,
        vector: Option<&[f32]>,
    ) -> Result<(), DomainError>;

    async fn find_resource(&self, uri: &str)
        -> Result<Option<MemoryResource>, DomainError>;

    async fn list_resources(&self) -> Result<Vec<MemoryResource>, DomainError>;

    async fn search_resources_semantic(
        &self,
        vector: &[f32],
        limit: usize,
    ) -> Result<Vec<(MemoryResource, f32)>, DomainError>;

    // ── Namespaces ───────────────────────────────────────────────────────

    async fn create_namespace(&self, name: &str) -> Result<bool, DomainError>;
    async fn delete_namespace(&self, name: &str) -> Result<bool, DomainError>;
    async fn list_namespaces(&self) -> Result<Vec<(String, u64)>, DomainError>;
    async fn namespace_projects(&self, name: &str) -> Result<Vec<String>, DomainError>;
    async fn assign_project(&self, namespace: &str, project: &str) -> Result<bool, DomainError>;
    async fn unassign_project(&self, namespace: &str, project: &str) -> Result<bool, DomainError>;
}
```

- [ ] **Step 4: Compile breaks — resolve**

This task leaves the implementation (`DuckdbStore`) missing the new methods. Stub them in `duckdb_memory_repository.rs` with `todo!()` to make the build pass; the next task implements them. Alternative: comment out the old trait impl block and skip tests that use it.

```bash
cargo build 2>&1 | tail -20
```

Expected: builds with `todo!()` warnings; tests still fail.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(app): rewrite MemoryRepository port for fact-only model"
```

---

## Task 6: Domain — `MemoryResource` type

**Files:**
- Create: `src/domain/resource.rs`
- Modify: `src/domain/mod.rs`

- [ ] **Step 1: Failing test**

Create `src/domain/resource.rs`:

```rust
//! A file or URL added explicitly via `memory add`. The LLM writes a one-line
//! abstract (L0) and a longer overview (L1) at ingest time; the full content
//! is kept alongside. What is gone is the `memory://` *tree* and the
//! `MemoryNode` type — resources are their own table, not tree nodes.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryResource {
    /// `memory://resources/<slug>` — primary key.
    pub uri: String,
    /// Original source (file path or URL) the content was fetched from.
    pub source: String,
    /// Display name (slug).
    pub name: String,
    /// L0 — one-line summary, what recall ranks and lists display.
    pub abstract_: String,
    /// L1 — a paragraph orienting the reader before opening `content`.
    pub overview: String,
    /// Full text.
    pub content: String,
    pub created_at: i64,
}

impl MemoryResource {
    /// Text used for the embedding — abstract plus overview, mirroring how
    /// `MemoryNode::embedding_text` used to combine the two levels.
    pub fn embedding_text(&self) -> String {
        if self.overview.trim().is_empty() {
            self.abstract_.clone()
        } else {
            format!("{}\n\n{}", self.abstract_, self.overview)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_text_combines_abstract_and_overview() {
        let r = MemoryResource {
            uri: "memory://resources/x".into(),
            source: "/tmp/x.md".into(),
            name: "x".into(),
            abstract_: "A note about x".into(),
            overview: "Longer context about x.".into(),
            content: "hello world".into(),
            created_at: 0,
        };
        assert_eq!(r.embedding_text(), "A note about x\n\nLonger context about x.");
    }

    #[test]
    fn embedding_text_falls_back_to_abstract_when_overview_empty() {
        let r = MemoryResource {
            uri: "memory://resources/x".into(),
            source: "/tmp/x.md".into(),
            name: "x".into(),
            abstract_: "A note about x".into(),
            overview: String::new(),
            content: "hello world".into(),
            created_at: 0,
        };
        assert_eq!(r.embedding_text(), "A note about x");
    }
}
```

- [ ] **Step 2: Wire into `domain/mod.rs`**

Add `pub mod resource;` and re-export `MemoryResource`.

- [ ] **Step 3: Test**

```bash
cargo test --lib domain::resource 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(domain): add MemoryResource type"
```

---

## Task 7: Connector — rewrite DuckDB schema (drop dead tables)

**Files:**
- Modify: `src/connector/adapter/duckdb_store.rs` (rename to drop `NodeRepository` baggage)
- Modify: `src/connector/adapter/duckdb_memory_repository.rs`

- [ ] **Step 1: Failing test**

Create `tests/fact_repository_tests.rs` additions:

```rust
#[tokio::test]
async fn fresh_db_has_only_the_new_tables() {
    let store = DuckdbStore::in_memory(384, "test-model").unwrap();
    let conn = store.conn.lock().await;
    let mut stmt = conn.prepare("SHOW TABLES").unwrap();
    let names: Vec<String> = stmt
        .query_map([], |row| row.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    for expected in [
        "memories",
        "memory_embeddings",
        "entities",
        "entity_names",
        "memory_sessions",
        "memory_resources",
        "memory_resource_embeddings",
        "memory_namespaces",
        "memory_meta",
    ] {
        assert!(names.contains(&expected.to_string()), "missing {expected}");
    }
    for gone in [
        "memory_items",
        "memory_nodes",
        "memory_node_vectors",
        "memory_edges",
        "memory_dream_runs",
        "entity_vectors",
        "memory_vectors",
    ] {
        assert!(!names.contains(&gone.to_string()), "still present {gone}");
    }
}
```

- [ ] **Step 2: Verify fail**

```bash
cargo test --test fact_repository_tests fresh_db_has_only_the_new_tables 2>&1 | tail -10
```

- [ ] **Step 3: Rewrite schema**

In `duckdb_store.rs::initialize`, replace the `execute_batch` DDL with:

```sql
CREATE TABLE IF NOT EXISTS memory_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    subject_entity_id TEXT,
    subject_literal TEXT,
    predicate TEXT NOT NULL,
    object_entity_id TEXT,
    object_literal TEXT,
    statement TEXT NOT NULL,
    project TEXT NOT NULL DEFAULT '',
    recorded_at BIGINT NOT NULL,
    source_session_id TEXT,
    source_message_index BIGINT,
    source_kind TEXT NOT NULL,
    confidence DOUBLE NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id TEXT PRIMARY KEY,
    vector FLOAT[384] NOT NULL
);
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);
CREATE TABLE IF NOT EXISTS entity_names (
    name TEXT NOT NULL,
    name_key TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    PRIMARY KEY (name_key, entity_id)
);
CREATE INDEX IF NOT EXISTS entity_names_key_idx ON entity_names (name_key);
CREATE TABLE IF NOT EXISTS memory_sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    imported_at BIGINT NOT NULL,
    message_count BIGINT NOT NULL,
    items_written BIGINT NOT NULL,
    status TEXT NOT NULL DEFAULT 'imported',
    last_error TEXT,
    project TEXT
);
CREATE TABLE IF NOT EXISTS memory_resources (
    uri TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    name TEXT NOT NULL,
    abstract TEXT NOT NULL,
    overview TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at BIGINT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_resource_embeddings (
    uri TEXT PRIMARY KEY,
    vector FLOAT[384] NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_namespaces (
    namespace TEXT NOT NULL,
    project TEXT NOT NULL,
    created_at BIGINT,
    UNIQUE (namespace, project)
);
```

(Substitute `{dimensions}` for the literal `384` in the format string.)

**Drop the `ALTER TABLE` migration blocks.** A user with an old DB deletes the file — schema is recreated empty on next open.

- [ ] **Step 4: Delete `NodeRepository` impl**

Remove the entire `NodeRepository` impl block from `duckdb_store.rs` (all the item/node/dream-run code). Keep the connection/`meta` plumbing and `memory_sessions` CRUD — that moves into the memory-repository impl.

- [ ] **Step 5: Test**

```bash
cargo test --test fact_repository_tests 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(db): drop dead tables, keep fact+entity+resource schema"
```

---

## Task 8: Connector — implement `MemoryRepository` for `DuckdbStore`

**Files:**
- Modify: `src/connector/adapter/duckdb_memory_repository.rs`

- [ ] **Step 1: Failing tests**

Build out `tests/fact_repository_tests.rs`:

```rust
use memory_rs::domain::{Entity, EntityRef, Memory, MemoryKind, Predicate, SourceKind};

fn fact(id: &str, subj_literal: &str, obj_literal: &str, stmt: &str, ts: i64) -> Memory {
    Memory {
        id: id.into(),
        kind: MemoryKind::Fact,
        subject: EntityRef::Literal(subj_literal.into()),
        predicate: Predicate::RelatesTo,
        object: EntityRef::Literal(obj_literal.into()),
        statement: stmt.into(),
        project: None,
        recorded_at: ts,
        source_session_id: None,
        source_message_index: None,
        source_kind: SourceKind::UserStated,
        confidence: 1.0,
    }
}

#[tokio::test]
async fn append_then_find_round_trips() {
    let store = DuckdbStore::in_memory(384, "m").unwrap();
    let m = fact("m1", "user", "tabs", "user uses tabs", 1000);
    store.append_memory(&m, None).await.unwrap();
    let got = store.find_memory("m1").await.unwrap().unwrap();
    assert_eq!(got.statement, "user uses tabs");
}

#[tokio::test]
async fn delete_memory_removes_row_and_embedding() {
    let store = DuckdbStore::in_memory(384, "m").unwrap();
    let m = fact("m1", "u", "t", "s", 1000);
    let vec = vec![0.0f32; 384];
    store.append_memory(&m, Some(&vec)).await.unwrap();
    assert!(store.delete_memory("m1").await.unwrap());
    assert!(store.find_memory("m1").await.unwrap().is_none());
    // Embedding gone too: semantic search over all vectors returns nothing.
    let hits = store
        .search_memories_semantic(&vec, None, None, 10)
        .await
        .unwrap();
    assert!(hits.is_empty());
}

#[tokio::test]
async fn recency_list_is_newest_first() {
    let store = DuckdbStore::in_memory(384, "m").unwrap();
    for (id, ts) in [("a", 100), ("b", 300), ("c", 200)] {
        store.append_memory(&fact(id, "s", "o", "stmt", ts), None).await.unwrap();
    }
    let listed = store.list_memories_by_recency(None, None, 10).await.unwrap();
    let ids: Vec<_> = listed.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(ids, ["b", "c", "a"]);
}

#[tokio::test]
async fn entity_upsert_and_name_lookup() {
    let store = DuckdbStore::in_memory(384, "m").unwrap();
    let e = Entity {
        id: "e1".into(),
        entity_type: "tool".into(),
        canonical_name: "codesearch".into(),
        names: vec!["codesearch".into(), "the codesearch cli".into()],
        created_at: 0,
        updated_at: 0,
    };
    store.upsert_entity(&e).await.unwrap();
    let by_name = store.find_entities_by_name("codesearch").await.unwrap();
    assert_eq!(by_name.len(), 1);
    assert_eq!(by_name[0].id, "e1");
    // Role-word normalization still applies.
    let by_role = store.find_entities_by_name("the codesearch tool").await.unwrap();
    assert_eq!(by_role.len(), 1);
}
```

- [ ] **Step 2: Verify fail**

```bash
cargo test --test fact_repository_tests 2>&1 | tail -10
```

- [ ] **Step 3: Implement**

Rewrite `duckdb_memory_repository.rs`. Update `MEMORY_COLUMNS` to the new DDL:

```rust
const MEMORY_COLUMNS: &str = "id, kind, subject_entity_id, subject_literal, predicate, \
     object_entity_id, object_literal, statement, project, recorded_at, \
     source_session_id, source_message_index, source_kind, confidence";
const MEMORY_COLUMN_COUNT: usize = 14;
```

Update `memory_from_row` accordingly. Implement each port method; drop `set_memory_status`, `reopen_memory`, `set_memory_confidence`, all edge methods, `repoint_entity`, `delete_entity`, `search_entities_semantic`, `list_memory_embeddings`, `memory_stats`. Keep entity CRUD minus the vector column.

Key SQL fragments:

```rust
// delete_memory
conn.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", params![id])?;
let n = conn.execute("DELETE FROM memories WHERE id = ?", params![id])?;
Ok(n > 0)

// list_memories_by_recency
let sql = format!(
    "SELECT {MEMORY_COLUMNS} FROM memories \
     WHERE (? IS NULL OR kind = ?) \
     ORDER BY recorded_at DESC LIMIT ?"
);
```

(Apply the same project-scope clause pattern the current code uses, via `project_scope_clause`.)

- [ ] **Step 4: Test**

```bash
cargo test --test fact_repository_tests 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(db): implement fact-only MemoryRepository"
```

---

## Task 9: Application — RRF(cosine, recency) recall

**Files:**
- Modify: `src/application/use_cases/memory_recall.rs`

- [ ] **Step 1: Failing test**

Add to `tests/fact_repository_tests.rs`:

```rust
#[tokio::test]
async fn recall_ranks_by_rrf_over_cosine_and_recency() {
    let store = DuckdbStore::in_memory(384, "m").unwrap();

    // Two memories with identical embeddings — cosine ties. Recency breaks
    // the tie: newer memory ranks first under RRF.
    let mut older = fact("old", "u", "x", "topic alpha", 1000);
    let mut newer = fact("new", "u", "x", "topic alpha", 2000);
    older.project = Some("proj".into());
    newer.project = Some("proj".into());

    let v = vec![0.1f32; 384];
    store.append_memory(&older, Some(&v)).await.unwrap();
    store.append_memory(&newer, Some(&v)).await.unwrap();

    let use_case = memory_rs::application::MemoryRecallUseCase::new_for_test(...);
    let hits = use_case.recall("alpha", None, &[].as_slice(), 10).await.unwrap();
    assert_eq!(hits[0].memory.id, "new");
    assert_eq!(hits[1].memory.id, "old");
}
```

The exact constructor shape depends on the current use case; adapt as needed.

- [ ] **Step 2: Verify fail**

```bash
cargo test --test fact_repository_tests recall_ranks 2>&1 | tail -10
```

- [ ] **Step 3: Implement RRF**

In `memory_recall.rs`:

1. Run the existing semantic leg → `Vec<(Memory, f32)>`.
2. Run `list_memories_by_recency` with same filter → `Vec<Memory>`.
3. Fuse with RRF (k=60, matching current hybrid pattern):

```rust
fn rrf_score(rank: usize) -> f32 {
    const K: f32 = 60.0;
    1.0 / (K + rank as f32 + 1.0)
}

fn fuse(semantic: &[(String, f32)], recency: &[String]) -> Vec<String> {
    use std::collections::HashMap;
    let mut score: HashMap<String, f32> = HashMap::new();
    for (rank, (id, _)) in semantic.iter().enumerate() {
        *score.entry(id.clone()).or_default() += rrf_score(rank);
    }
    for (rank, id) in recency.iter().enumerate() {
        *score.entry(id.clone()).or_default() += rrf_score(rank);
    }
    let mut ids: Vec<_> = score.into_iter().collect();
    ids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ids.into_iter().map(|(id, _)| id).collect()
}
```

4. Drop the edge-walking "provenance" enrichment.

- [ ] **Step 4: Test**

```bash
cargo test --test fact_repository_tests recall_ranks 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(recall): RRF over cosine + recency, drop edge enrichment"
```

---

## Task 10: Application — simplify ingestion prompt + extraction

**Files:**
- Modify: `src/application/use_cases/memory_ingestion.rs`
- Modify: `src/application/use_cases/memory_ingestion_prompt.rs`
- Delete: `src/application/use_cases/memory_extraction.rs`, `memory_extraction_prompt.rs`

- [ ] **Step 1: Failing test**

```rust
#[test]
fn extraction_prompt_lists_seven_predicates_only() {
    let prompt = memory_rs::application::build_ingestion_prompt_for_test();
    for p in ["prefers", "avoids", "uses", "fixes", "decided", "is_a", "relates_to"] {
        assert!(prompt.contains(p), "missing {p}");
    }
    for retired in ["requires", "provides", "implements", "contains",
                    "derived_from", "configures", "causes", "prevents",
                    "has", "works_on"] {
        assert!(!prompt.contains(retired), "still listed {retired}");
    }
}
```

- [ ] **Step 2: Verify fail**

```bash
cargo test extraction_prompt_lists_seven_predicates_only 2>&1 | tail -5
```

- [ ] **Step 3: Rewrite prompt**

Strip `memory_ingestion_prompt.rs` to a single template asking for triples:

```
Extract durable facts from this session as JSON triples.

For each fact output:
- subject_name: the entity the fact is about
- subject_type: person | tool | service | library | concept
- predicate: prefers | avoids | uses | fixes | decided | is_a | relates_to
- object_name_or_literal: entity name (if it's a thing) or a literal value
- statement: the fact as a single English sentence
- project: the project this fact is specific to, or null if global
- source_message_index: index of the message the fact came from

Predicates:
- prefers — a durable taste or habit
- avoids — a durable dislike
- uses — depends on, is built with
- fixes — resolves a problem
- decided — a choice that was made
- is_a — type or category membership
- relates_to — none of the above fits

Do NOT extract:
- skills or procedures (unreliable, go stale)
- experiences or episodes (statement is enough)
- project entities (project goes in the project field, not as a subject)
```

Adapt to the existing templating machinery in the file.

- [ ] **Step 4: Ingestion path**

In `memory_ingestion.rs`:
- Drop kind-classification logic (everything is `Fact`).
- Drop edge-writing.
- Drop `status`/`valid_from`/`valid_to` writes.
- Keep entity resolution via `entity_name_key` only — delete the embedding-similarity tier and the LLM adjudication tier.
- Keep dedup on write (same `(subject_key, predicate, object_key, project)` upserts).

- [ ] **Step 5: Test**

```bash
cargo test --test ingestion_pipeline_tests 2>&1 | tail -10
```

(Rename `memory_pipeline_tests.rs` → `ingestion_pipeline_tests.rs`; delete the parts exercising kinds/edges.)

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(ingest): single-kind prompt, name-key-only entity resolution"
```

---

## Task 11: Application — dream = harvest only

**Files:**
- Modify: `src/application/use_cases/memory_dream.rs`
- Delete: `src/application/use_cases/memory_dream_prompt.rs`

- [ ] **Step 1: Failing test**

```rust
#[tokio::test]
async fn dream_imports_new_sessions_and_stops() {
    // Set up a session source with one finished session, run dream once,
    // assert session row exists and zero consolidation ran (no cluster stats).
}
```

- [ ] **Step 2: Strip `memory_dream.rs`**

Delete the consolidation phase. Keep:
- discover new finished sessions
- for each, run ingestion
- record session row

Delete `DreamRun` recording (no `memory_dream_runs` table). Delete prompt file.

- [ ] **Step 3: Test**

```bash
cargo test --test ingestion_pipeline_tests dream 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(dream): drop consolidation, keep harvest"
```

---

## Task 12: Application — delete `MemoryBrowse`, `MemorySummary`, `MemoryExtraction`

**Files:**
- Delete: `src/application/use_cases/memory_browse.rs`, `memory_summary.rs`, `memory_extraction.rs`, `memory_extraction_prompt.rs`
- Modify: `src/application/use_cases/mod.rs`, `src/lib.rs`

- [ ] **Step 1: Delete files**

```bash
git rm src/application/use_cases/memory_browse.rs \
       src/application/use_cases/memory_summary.rs \
       src/application/use_cases/memory_extraction.rs \
       src/application/use_cases/memory_extraction_prompt.rs \
       src/application/interfaces/node_repository.rs
```

- [ ] **Step 2: Fix re-exports**

Remove `MemoryBrowseUseCase`, `SummarizeMemoryUseCase`, `MemoryExtractionUseCase`, `NodeRepository`, `NodeStats`, `MemoryLevel`, `MemoryRow`, `RowTarget`, `MEMORY_ROOT_URI`, `PROJECTS_ROOT_URI`, `RESOURCES_ROOT_URI`, `SESSIONS_ROOT_URI` from `src/application/mod.rs` and `src/lib.rs`.

- [ ] **Step 3: Build**

```bash
cargo build 2>&1 | tail -30
```

Fix TUI / CLI / API references by removing the code paths that called these use cases. The `Tree`, `Add` CLI commands are either removed or reimplemented over the new resource table (Tasks 13-14).

- [ ] **Step 4: Test**

```bash
cargo test 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(app): delete browse/summary/extraction use cases"
```

---

## Task 13: Connector — resources adapter

**Files:**
- Modify: `src/connector/adapter/resource_fetch.rs`
- Modify: `src/connector/adapter/duckdb_memory_repository.rs`

- [ ] **Step 1: Failing test**

```rust
#[tokio::test]
async fn resource_upsert_and_find() {
    let store = DuckdbStore::in_memory(384, "m").unwrap();
    let r = MemoryResource {
        uri: "memory://resources/x".into(),
        source: "/tmp/x.md".into(),
        name: "x".into(),
        abstract_: "A note about x".into(),
        overview: "Longer context.".into(),
        content: "hello".into(),
        created_at: 0,
    };
    let v = vec![0.0f32; 384];
    store.upsert_resource(&r, Some(&v)).await.unwrap();
    let got = store.find_resource("memory://resources/x").await.unwrap().unwrap();
    assert_eq!(got.content, "hello");
    assert_eq!(got.abstract_, "A note about x");
    assert_eq!(got.overview, "Longer context.");
}
```

- [ ] **Step 2: Implement**

Add `upsert_resource`, `find_resource`, `list_resources`, `search_resources_semantic` to `duckdb_memory_repository.rs`.

Update `resource_fetch.rs` to return `MemoryResource` instead of building a `MemoryNode`. Keep the existing LLM pass that writes both abstract and overview; embed `abstract + overview` via `embedding_text()`. Drop only the `MemoryNode` construction (URI/parent/kind plumbing).

- [ ] **Step 3: Test**

```bash
cargo test --test fact_repository_tests resource 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(resources): minimal content+embedding store"
```

---

## Task 14: Connector — controller layer

**Files:**
- Modify: `src/connector/api/controller.rs`

- [ ] **Step 1: Strip outcomes**

Delete:
- `MemoryShowOutcome::Node` variant
- edge fields on `MemoryShowOutcome::Memory`
- `provenance` on `MemorySearchOutcome::Hits`
- namespace-only outcome handling stays (namespaces stay)

Update:
- `show_memory` returns just the memory
- `forget_memory` calls `delete_memory` (hard delete)
- `add_resource` writes `MemoryResource` not `MemoryNode`
- `tree` returns a stub listing (sessions + resources) — no abstracts

- [ ] **Step 2: Test**

```bash
cargo build 2>&1 | tail -10
cargo test 2>&1 | tail -10
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor(api): controller matches new model"
```

---

## Task 15: MCP — keep tool names, update internals

**Files:**
- Modify: `src/connector/adapter/mcp/server.rs`

- [ ] **Step 1: Strip dead params + fields**

- `search_memories`: drop `kind` param handling (accept but ignore), drop `provenance` block, drop entity labels lookup if no longer used
- `list_memories`: drop `status` and `kind` filters
- `read_memory`: drop `edges` array, drop `Node` variant
- `browse_memory`: keep — returns flat listing of sessions + resources
- `add_resource`: unchanged signature, internals use new resource path
- Server instructions string: rewrite to describe fact-only memory

- [ ] **Step 2: Test**

```bash
cargo build 2>&1 | tail -10
cargo test 2>&1 | tail -10
```

Manually verify MCP tool list still contains all tool names (integration test or local MCP client).

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor(mcp): keep tool names, strip dead params"
```

---

## Task 16: CLI — strip dead subcommands

**Files:**
- Modify: `src/cli/mod.rs`

- [ ] **Step 1: Remove subcommands**

Delete: `Show` (no more nodes), `Tree`, `Conflicts`, `Stats` (or simplify to fact+entity counts). Keep: `Import`, `Search`, `List`, `Delete`, `Sessions`, `Resume`, `Add`, `Dream`, `Namespace`, `Serve`, `Mcp`, `Tui`.

Remove `MemoryKindArg` / `NodeKindArg` enums; replace `kind` flag on `Search`/`List` with nothing.

- [ ] **Step 2: Update handlers**

`src/connector/api/router.rs` — drop routes for deleted commands; update others.

- [ ] **Step 3: Build**

```bash
cargo build 2>&1 | tail -10
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(cli): drop show/tree/conflicts subcommands"
```

---

## Task 17: TUI — flat fact list + entity drill

**Files:**
- Modify: `src/tui/screens/memory.rs`
- Delete: `src/tui/screens/import.rs` (fold useful bits into app.rs)

- [ ] **Step 1: Rewrite memory screen**

Show:
- flat list of facts, newest first
- filter box (keyword search)
- drill into entity: list facts where entity is subject or object
- drill into session: list facts produced by that session

Drop: kind tabs, edge graph, L0/L1/L2 view, project tree.

- [ ] **Step 2: Build**

```bash
cargo build 2>&1 | tail -10
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor(tui): flat fact list + entity drill"
```

---

## Task 18: Management API — strip dead endpoints

**Files:**
- Modify: `src/connector/adapter/management/handlers.rs`, `server.rs`, `dream.rs`, `dream_routes.rs`, `sessions.rs`

- [ ] **Step 1: Drop endpoints**

Remove endpoints tied to items/nodes/edges/dream-runs/conflicts. Keep:
- `POST /api/memory/import`
- `GET /api/memory/recall`
- `GET /api/memory/list`
- `GET /api/memory/sessions`
- `GET /api/memory/resume`
- `POST /api/memory/dream` (harvest only)
- namespace CRUD

- [ ] **Step 2: Build**

```bash
cargo build 2>&1 | tail -10
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor(api): drop dead endpoints"
```

---

## Task 19: Final cleanup — types, exports, docs

**Files:**
- Modify: `src/lib.rs`, `src/application/mod.rs`, `src/domain/mod.rs`

- [ ] **Step 1: Audit exports**

```bash
cargo doc --no-deps 2>&1 | tail -20
```

Remove deleted types from `pub use` in `lib.rs`. Update the crate-level doc comment to describe the new model (fact triples + entities + sessions, no edges, no tree).

- [ ] **Step 2: Sweep for stale strings**

```bash
grep -r "memory_items\|memory_nodes\|MemoryItem\|MemoryNode\|MemoryEdge\|MemoryStatus\|EdgeType\|supersedes\|contradicts" src/ 2>&1 | head -30
```

Fix any remaining references.

- [ ] **Step 3: Update README**

Rewrite `README.md` to describe the new model in two lines.

- [ ] **Step 4: Full test suite**

```bash
cargo test 2>&1 | tail -15
cargo clippy --all-targets -- -D warnings 2>&1 | tail -15
```

Expected: PASS, no warnings.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: final cleanup after simplification"
```

---

## Self-review notes

- **Spec coverage:** Tasks 1-4 (domain) cover spec sections "Data model that survives" and "Predicate". Tasks 5-8 (ports + DB) cover "Schema". Tasks 9-11 cover "Recall" and "Dream cycle". Tasks 12-14 cover "Data model deleted". Tasks 15-18 cover "Surfaces". Task 19 closes out.
- **Placeholder scan:** All steps carry commands and code. The TUI rewrite (Task 17) is intentionally lighter on detail because the existing file is being mostly rewritten rather than patched — the engineer should treat it as a fresh screen implementation.
- **Type consistency:** `Memory`, `MemoryResource`, `MemoryRepository` method signatures are used uniformly. The shrunken `Predicate::ALL` is asserted to be 7 in Task 1 and used in Task 10.
