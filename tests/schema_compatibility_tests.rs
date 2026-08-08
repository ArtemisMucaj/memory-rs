//! Schema-compatibility tests: opening an *existing* `memory.duckdb` with a
//! build that knows about memories.
//!
//! The memory graph was added by folding its tables into the same
//! `CREATE TABLE IF NOT EXISTS` batch the item store already ran, rather than
//! by a versioned migration. That is the cheapest thing that works, but it puts
//! the entire backward-compatibility story in one place: the batch runs on
//! *every* open, including the first open of a database written before memories
//! existed. If it were ever non-idempotent — an `ALTER`, a bare `CREATE`, a
//! stray `INSERT` — the failure would land on a user's real store on upgrade,
//! which is the worst possible place to discover it.
//!
//! So these tests do the one thing the unit tests cannot: they hand-build a
//! database file with the *old* DDL, then open it with the current code and
//! assert the old data survived and the new tables appeared. `in_memory` stores
//! start empty and can never catch this.

use duckdb::Connection;
use memory_rs::application::MemoryRepository;
use memory_rs::domain::{EntityRef, Memory, MemoryKind, MemoryStatus, SourceKind};
use memory_rs::Predicate;
use memory_rs::{DuckdbStore, NodeRepository};

const DIMS: usize = 4;

/// The schema exactly as it stood before the memory layer, so the file on disk
/// is byte-for-byte what the previous release would have written — including a
/// populated item, session and namespace to prove nothing is dropped.
const PRE_MEMORY_DDL: &str = r#"
    CREATE TABLE IF NOT EXISTS memory_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS memory_items (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        name TEXT NOT NULL,
        content TEXT NOT NULL,
        source_session_id TEXT,
        project TEXT NOT NULL DEFAULT '',
        created_at BIGINT NOT NULL,
        updated_at BIGINT NOT NULL,
        update_count BIGINT NOT NULL DEFAULT 0,
        UNIQUE (kind, name, project)
    );
    CREATE TABLE IF NOT EXISTS memory_vectors (
        item_id TEXT PRIMARY KEY,
        vector FLOAT[4] NOT NULL
    );
    CREATE TABLE IF NOT EXISTS memory_sessions (
        id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        imported_at BIGINT NOT NULL,
        message_count BIGINT NOT NULL,
        items_written BIGINT NOT NULL,
        status TEXT NOT NULL DEFAULT 'imported',
        last_error TEXT
    );
    CREATE TABLE IF NOT EXISTS memory_nodes (
        uri TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        parent_uri TEXT,
        label TEXT,
        abstract TEXT NOT NULL,
        overview TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at BIGINT NOT NULL,
        updated_at BIGINT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS memory_node_vectors (
        node_uri TEXT PRIMARY KEY,
        vector FLOAT[4] NOT NULL
    );
    CREATE TABLE IF NOT EXISTS memory_dream_runs (
        id TEXT PRIMARY KEY,
        started_at BIGINT NOT NULL,
        finished_at BIGINT NOT NULL,
        sessions_imported BIGINT NOT NULL,
        clusters_found BIGINT NOT NULL,
        operations_applied BIGINT NOT NULL,
        operations_skipped BIGINT NOT NULL,
        status TEXT NOT NULL DEFAULT 'completed'
    );
    CREATE TABLE IF NOT EXISTS memory_namespaces (
        namespace TEXT NOT NULL,
        project TEXT NOT NULL,
        UNIQUE (namespace, project)
    );
    INSERT INTO memory_meta (key, value) VALUES ('dimensions', '4');
    INSERT INTO memory_meta (key, value) VALUES ('embedding_model', 'model-old');
    INSERT INTO memory_items
        (id, kind, name, content, source_session_id, project, created_at, updated_at, update_count)
        VALUES ('item-1', 'fact', 'legacy_fact', 'the old content', 'session-1', '', 100, 100, 0);
    INSERT INTO memory_vectors (item_id, vector) VALUES ('item-1', [1.0, 0.0, 0.0, 0.0]::FLOAT[4]);
    INSERT INTO memory_sessions (id, source, imported_at, message_count, items_written, status, last_error)
        VALUES ('session-1', 'claude', 100, 3, 1, 'imported', NULL);
    INSERT INTO memory_namespaces (namespace, project) VALUES ('ns-a', 'owner/repo');
"#;

fn pre_memory_database(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("memory.duckdb");
    let conn = Connection::open(&path).unwrap();
    conn.execute_batch(PRE_MEMORY_DDL).unwrap();
    drop(conn);
    path
}

fn sample_memory(id: &str) -> Memory {
    Memory {
        id: id.to_string(),
        kind: MemoryKind::Fact,
        subject: EntityRef::Entity("entity-1".to_string()),
        predicate: Predicate::Uses,
        object: EntityRef::Literal("svc-a".to_string()),
        statement: "the team uses svc-a".to_string(),
        project: Some("owner/repo".to_string()),
        recorded_at: 1_700_000_000,
        valid_from: 1_700_000_000,
        valid_to: None,
        source_session_id: Some("session-1".to_string()),
        source_message_index: Some(3),
        source_kind: SourceKind::UserStated,
        confidence: 0.9,
        status: MemoryStatus::Active,
        derived: false,
        derived_from: Vec::new(),
    }
}

/// Upgrading a populated pre-memory store must be purely additive, and must stay
/// additive across repeated opens — the batch runs every time, not once.
#[tokio::test]
async fn opening_a_pre_memory_database_is_additive() {
    let dir = tempfile::tempdir().unwrap();
    let path = pre_memory_database(dir.path());

    // First open with the memory-aware build.
    let repo = DuckdbStore::new(&path, DIMS, "model-new").unwrap();
    // The stored embedding model still wins over the configured one: a config
    // change must not strand a store whose vectors were written by another
    // model. This is why the memory layer reuses `memory_meta` rather than
    // adding a `memory_meta` with its own hard check.
    assert_eq!(repo.stored_embedding_model(), "model-old");

    let items = NodeRepository::list_items(&repo, None).await.unwrap();
    assert_eq!(items.len(), 1, "the legacy item was lost on upgrade");
    assert_eq!(items[0].name(), "legacy_fact");
    let sessions = NodeRepository::list_sessions(&repo).await.unwrap();
    assert_eq!(sessions.len(), 1, "the legacy session was lost on upgrade");
    // `memory_sessions.project` is added by an `ALTER` on upgrade. A legacy row
    // has none, and that has to read as "unknown" rather than fail the open —
    // this is the column the resume briefing scopes on.
    assert_eq!(
        sessions[0].project, None,
        "a pre-project session must migrate to an unattributed one",
    );
    assert_eq!(
        NodeRepository::list_namespaces(&repo).await.unwrap().len(),
        1,
        "the legacy namespace mapping was lost on upgrade"
    );
    // `memory_namespaces.created_at` is added by an `ALTER` on upgrade and is
    // deliberately left NULL: a legacy namespace has no recorded creation date,
    // so it yields no auto-import cutoff and harvests nothing. Backfilling it
    // to epoch 0 would make the first dream after an upgrade try to import the
    // machine's entire session history — the exact failure the cutoff prevents.
    assert!(
        NodeRepository::namespaced_project_cutoffs(&repo)
            .await
            .unwrap()
            .is_empty(),
        "a dateless legacy namespace must not become auto-importable on upgrade"
    );

    // The memory tables now exist on a file that was written without them.
    MemoryRepository::append_memory(
        &repo,
        &sample_memory("memory-1"),
        Some(&[0.1, 0.2, 0.3, 0.4]),
    )
    .await
    .unwrap();
    drop(repo);

    // Second open: the batch runs again, now over a file that already has both
    // halves. Nothing may be recreated, truncated or duplicated.
    let repo = DuckdbStore::new(&path, DIMS, "model-new").unwrap();
    assert_eq!(
        NodeRepository::list_items(&repo, None).await.unwrap().len(),
        1
    );
    assert_eq!(
        MemoryRepository::find_memory(&repo, "memory-1")
            .await
            .unwrap()
            .map(|c| c.statement),
        Some("the team uses svc-a".to_string())
    );
    assert_eq!(
        MemoryRepository::memory_stats(&repo)
            .await
            .unwrap()
            .total_memories,
        1
    );
    drop(repo);

    // Third open, to catch anything that accumulates only after the second.
    let repo = DuckdbStore::new(&path, DIMS, "model-new").unwrap();
    assert_eq!(
        MemoryRepository::list_memories(&repo, None, None, None)
            .await
            .unwrap()
            .len(),
        1
    );
}

/// A store old enough to predate the `(kind, name)` -> `(kind, name, project)`
/// widening, so the item-identity migration fires in the *same* open as the
/// memory DDL. The two run against one connection and must not interfere.
#[tokio::test]
async fn legacy_identity_migration_still_runs_alongside_the_memory_ddl() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.duckdb");
    let conn = Connection::open(&path).unwrap();
    conn.execute_batch(
        r#"
        CREATE TABLE memory_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE memory_items (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            content TEXT NOT NULL,
            source_session_id TEXT,
            project TEXT,
            created_at BIGINT NOT NULL,
            updated_at BIGINT NOT NULL,
            update_count BIGINT NOT NULL DEFAULT 0,
            UNIQUE (kind, name)
        );
        CREATE TABLE memory_vectors (item_id TEXT PRIMARY KEY, vector FLOAT[4] NOT NULL);
        CREATE TABLE memory_sessions (
            id TEXT PRIMARY KEY, source TEXT NOT NULL, imported_at BIGINT NOT NULL,
            message_count BIGINT NOT NULL, items_written BIGINT NOT NULL
        );
        CREATE TABLE memory_nodes (
            uri TEXT PRIMARY KEY, kind TEXT NOT NULL, parent_uri TEXT,
            abstract TEXT NOT NULL, overview TEXT NOT NULL, content TEXT NOT NULL,
            created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL
        );
        INSERT INTO memory_meta (key, value) VALUES ('dimensions', '4');
        INSERT INTO memory_items
            (id, kind, name, content, source_session_id, project, created_at, updated_at, update_count)
            VALUES ('item-1', 'fact', 'legacy_fact', 'old content', NULL, NULL, 100, 100, 0);
        "#,
    )
    .unwrap();
    drop(conn);

    let repo = DuckdbStore::new(&path, DIMS, "model-new").unwrap();
    let items = NodeRepository::list_items(&repo, None).await.unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].project(), None);
    // And the memory half of the schema came up in that same open.
    MemoryRepository::append_memory(&repo, &sample_memory("memory-1"), None)
        .await
        .unwrap();
}
