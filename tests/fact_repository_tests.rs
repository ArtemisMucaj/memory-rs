//! End-to-end tests for the fact+entity store.
//!
//! Replaces the pre-simplification suites (`duckdb_store_tests`,
//! `duckdb_memory_repository_tests`, `memory_pipeline_tests`,
//! `import_pipeline_tests`, `schema_compatibility_tests`, `resume_tests`),
//! which exercised the deleted `MemoryItem`/`MemoryNode`/`MemoryEdge`
//! surfaces.

mod common;

use memory_rs::application::MemoryRepository;
use memory_rs::connector::DuckdbStore;
use memory_rs::domain::{
    Entity, ImportedSession, Memory, MemoryKind, MemoryResource, SessionStatus, SourceKind,
};

fn fact(id: &str, statement: &str, recorded_at: i64) -> Memory {
    Memory {
        id: id.into(),
        kind: MemoryKind::Fact,
        statement: statement.into(),
        entity_ids: Vec::new(),
        project: None,
        recorded_at,
        source_session_id: None,
        source_message_index: None,
        source_kind: SourceKind::UserStated,
        confidence: 1.0,
    }
}

#[tokio::test]
async fn append_then_find_round_trips() {
    let store = DuckdbStore::in_memory(common::DIMS, "test-model").unwrap();
    let m = fact("m1", "user prefers tabs", 1000);
    store.append_memory(&m, None).await.unwrap();
    let got = store.find_memory("m1").await.unwrap().unwrap();
    assert_eq!(got.statement, "user prefers tabs");
    assert_eq!(got.kind, MemoryKind::Fact);
}

#[tokio::test]
async fn delete_memory_removes_row_and_embedding() {
    let store = DuckdbStore::in_memory(common::DIMS, "test-model").unwrap();
    let m = fact("m1", "user prefers tabs", 1000);
    let v = vec![0.0f32; common::DIMS];
    store.append_memory(&m, Some(&v)).await.unwrap();
    assert!(store.delete_memory("m1").await.unwrap());
    assert!(store.find_memory("m1").await.unwrap().is_none());
    // Embedding gone too: semantic search returns nothing.
    let hits = store
        .search_memories_semantic(&v, None, None, 10)
        .await
        .unwrap();
    assert!(hits.is_empty());
}

#[tokio::test]
async fn recency_list_is_newest_first() {
    let store = DuckdbStore::in_memory(common::DIMS, "test-model").unwrap();
    for (id, ts) in [("a", 100), ("b", 300), ("c", 200)] {
        store.append_memory(&fact(id, "s", ts), None).await.unwrap();
    }
    let listed = store
        .list_memories_by_recency(None, None, 10)
        .await
        .unwrap();
    let ids: Vec<_> = listed.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(ids, ["b", "c", "a"]);
}

#[tokio::test]
async fn entity_upsert_and_name_lookup() {
    let store = DuckdbStore::in_memory(common::DIMS, "test-model").unwrap();
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
    let by_role = store
        .find_entities_by_name("the codesearch cli")
        .await
        .unwrap();
    assert_eq!(by_role.len(), 1);
}

#[tokio::test]
async fn project_is_not_an_entity_type() {
    use memory_rs::domain::VALID_ENTITY_TYPES;
    assert!(!VALID_ENTITY_TYPES.contains(&"project"));
}

#[tokio::test]
async fn resource_round_trips_with_abstract_and_overview() {
    let store = DuckdbStore::in_memory(common::DIMS, "test-model").unwrap();
    let r = MemoryResource {
        uri: "memory://resources/x".into(),
        source: "/tmp/x.md".into(),
        name: "x".into(),
        abstract_: "A note about x".into(),
        overview: "Longer context.".into(),
        content: "hello".into(),
        created_at: 0,
    };
    let v = vec![0.0f32; common::DIMS];
    store.upsert_resource(&r, Some(&v)).await.unwrap();
    let got = store
        .find_resource("memory://resources/x")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(got.content, "hello");
    assert_eq!(got.abstract_, "A note about x");
    assert_eq!(got.overview, "Longer context.");
}

#[tokio::test]
async fn sessions_record_and_list() {
    let store = DuckdbStore::in_memory(common::DIMS, "test-model").unwrap();
    let s = ImportedSession {
        id: "s1".into(),
        source: "claude:/tmp/x.jsonl".into(),
        imported_at: 100,
        message_count: 10,
        project: Some("memory-rs".into()),
        items_written: 3,
        status: SessionStatus::Imported,
        last_error: None,
    };
    store.record_session(&s).await.unwrap();
    let got = store
        .find_session("claude:/tmp/x.jsonl", "s1")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(got.id, "s1");
    assert_eq!(got.project.as_deref(), Some("memory-rs"));

    let listed = store.list_sessions(None, 10).await.unwrap();
    assert_eq!(listed.len(), 1);
}

#[tokio::test]
async fn namespaces_round_trip() {
    let store = DuckdbStore::in_memory(common::DIMS, "test-model").unwrap();
    assert!(store.create_namespace("work").await.unwrap());
    // Second create is idempotent.
    assert!(!store.create_namespace("work").await.unwrap());
    assert!(store.assign_project("work", "memory-rs").await.unwrap());
    assert!(!store.assign_project("work", "memory-rs").await.unwrap());
    let projects = store.namespace_projects("work").await.unwrap();
    assert_eq!(projects, vec!["memory-rs".to_string()]);
    let list = store.list_namespaces().await.unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].0, "work");
    assert_eq!(list[0].1, 1);
    assert!(store.namespace_created_at("work").await.unwrap().is_some());
    assert!(store.unassign_project("work", "memory-rs").await.unwrap());
    assert!(!store.unassign_project("work", "memory-rs").await.unwrap());
}

#[tokio::test]
async fn sessions_with_colliding_ids_from_different_sources_coexist() {
    // Claude, OpenCode and Zed mint ids from independent namespaces. The
    // composite `(source, id)` key is what keeps two sources reusing the
    // same session id from clobbering each other's markers.
    let store = DuckdbStore::in_memory(common::DIMS, "test-model").unwrap();
    for source in ["claude:proj/s.jsonl", "opencode:s", "zed:s"] {
        let s = ImportedSession {
            id: "s".into(),
            source: source.into(),
            imported_at: 100,
            message_count: 10,
            project: None,
            items_written: 0,
            status: SessionStatus::Imported,
            last_error: None,
        };
        store.record_session(&s).await.unwrap();
    }
    let listed = store.list_sessions(None, 10).await.unwrap();
    assert_eq!(listed.len(), 3, "each source keeps its own marker");
}

#[tokio::test]
async fn legacy_schema_is_rejected_at_open() {
    // Build a store with the pre-simplification shape: a `memories` table
    // carrying `predicate` / `subject_entity_id` / `object_*`. Opening it
    // with the new build must fail with the wipe-and-reimport message, not
    // at first write with a type error.
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("memory.duckdb");
    {
        let conn = duckdb::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                subject_entity_id TEXT,
                subject_literal TEXT,
                predicate TEXT NOT NULL,
                object_entity_id TEXT,
                object_literal TEXT,
                statement TEXT NOT NULL
            );",
        )
        .unwrap();
    }
    let err = DuckdbStore::new(&db_path, common::DIMS, "test-model")
        .err()
        .expect("legacy store must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("older version") && msg.contains("Delete `memory.duckdb`"),
        "unexpected error: {msg}"
    );
}

#[tokio::test]
async fn fresh_db_has_only_the_new_tables() {
    let store = DuckdbStore::in_memory(common::DIMS, "test-model").unwrap();
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
