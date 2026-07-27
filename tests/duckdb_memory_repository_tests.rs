//! Storage-layer integration tests for [`DuckdbMemoryRepository`].
//!
//! These drive the DuckDB adapter directly — no LLM, no network — to verify
//! the `FLOAT[dimensions]` vector round-trip, native `array_cosine_distance`
//! semantic recall, keyword search, the `(kind, name, project)` item identity,
//! and node / session / dream-run persistence. An in-memory database and small
//! hand-written vectors keep every test deterministic and fast.

use std::sync::Arc;

use memory_rs::{
    DreamRun, DuckdbMemoryRepository, ImportedSession, MemoryItem, MemoryKind, MemoryNode,
    MemoryRepository, NodeKind, SessionStatus,
};

const DIMS: usize = 4;

fn repo() -> Arc<dyn MemoryRepository> {
    Arc::new(DuckdbMemoryRepository::in_memory(DIMS, "mock-embedding").unwrap())
}

fn item(name: &str, content: &str, project: Option<&str>) -> MemoryItem {
    MemoryItem::new(
        format!("id-{name}-{}", project.unwrap_or("global")),
        MemoryKind::Fact,
        name.to_string(),
        content.to_string(),
        None,
        project.map(str::to_string),
        1,
        1,
        0,
    )
}

#[tokio::test]
async fn item_round_trips_by_identity_and_id() {
    let repo = repo();
    let it = item(
        "flaky_test_fix",
        "retry the timing-sensitive assertion",
        None,
    );
    repo.upsert_item(&it, None).await.unwrap();

    let by_identity = repo
        .find_item(MemoryKind::Fact, "flaky_test_fix", None)
        .await
        .unwrap()
        .expect("item should be found by identity");
    assert_eq!(
        by_identity.content(),
        "retry the timing-sensitive assertion"
    );

    let by_id = repo
        .find_item_by_id(it.id())
        .await
        .unwrap()
        .expect("item should be found by id");
    assert_eq!(by_id.name(), "flaky_test_fix");
}

#[tokio::test]
async fn same_name_in_two_projects_stays_two_items() {
    let repo = repo();
    repo.upsert_item(&item("build", "run make in svc-a", Some("svc-a")), None)
        .await
        .unwrap();
    repo.upsert_item(&item("build", "run cargo in svc-b", Some("svc-b")), None)
        .await
        .unwrap();

    let a = repo
        .find_item(MemoryKind::Fact, "build", Some("svc-a"))
        .await
        .unwrap()
        .unwrap();
    let b = repo
        .find_item(MemoryKind::Fact, "build", Some("svc-b"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(a.content(), "run make in svc-a");
    assert_eq!(b.content(), "run cargo in svc-b");

    // A global item of the same name is a third, distinct item.
    assert!(repo
        .find_item(MemoryKind::Fact, "build", None)
        .await
        .unwrap()
        .is_none());

    let named = repo
        .find_items_named(MemoryKind::Fact, "build")
        .await
        .unwrap();
    assert_eq!(named.len(), 2);
}

#[tokio::test]
async fn semantic_search_ranks_by_cosine_distance() {
    let repo = repo();
    let near = item("near", "aligned with the query", None);
    let far = item("far", "pointing the other way", None);
    // Query is [1,0,0,0]; `near` is close, `far` is orthogonal/opposite.
    repo.upsert_item(&near, Some(&[0.9, 0.1, 0.0, 0.0]))
        .await
        .unwrap();
    repo.upsert_item(&far, Some(&[0.0, 1.0, 0.0, 0.0]))
        .await
        .unwrap();

    let results = repo
        .search_semantic(&[1.0, 0.0, 0.0, 0.0], None, None, 10)
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0].0.name(),
        "near",
        "closest vector must rank first"
    );
    assert!(
        results[0].1 > results[1].1,
        "scores must be ordered: {} !> {}",
        results[0].1,
        results[1].1
    );
}

#[tokio::test]
async fn keyword_search_matches_without_embeddings() {
    let repo = repo();
    repo.upsert_item(
        &item(
            "timeout_handling",
            "retry network timeouts with backoff",
            None,
        ),
        None,
    )
    .await
    .unwrap();
    repo.upsert_item(&item("unrelated", "something else entirely", None), None)
        .await
        .unwrap();

    let hits = repo
        .search_keyword("network timeout", None, None, 10)
        .await
        .unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].0.name(), "timeout_handling");
}

#[tokio::test]
async fn item_vectors_round_trip_through_json() {
    let repo = repo();
    let it = item("vecd", "has a vector", None);
    let vec = [0.25, 0.5, 0.75, 1.0];
    repo.upsert_item(&it, Some(&vec)).await.unwrap();

    let one = repo
        .find_item_vector(it.id())
        .await
        .unwrap()
        .expect("vector should be stored");
    assert_eq!(one.len(), DIMS);
    for (got, want) in one.iter().zip(vec.iter()) {
        assert!((got - want).abs() < 1e-6, "{got} != {want}");
    }

    let all = repo.list_item_vectors().await.unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].0, it.id());
}

#[tokio::test]
async fn delete_removes_item_and_vector() {
    let repo = repo();
    let it = item("temp", "delete me", None);
    repo.upsert_item(&it, Some(&[1.0, 0.0, 0.0, 0.0]))
        .await
        .unwrap();

    assert!(repo
        .delete_item(MemoryKind::Fact, "temp", None)
        .await
        .unwrap());
    assert!(repo
        .find_item(MemoryKind::Fact, "temp", None)
        .await
        .unwrap()
        .is_none());
    assert!(repo.find_item_vector(it.id()).await.unwrap().is_none());
    // Deleting again is a no-op.
    assert!(!repo
        .delete_item(MemoryKind::Fact, "temp", None)
        .await
        .unwrap());
}

#[tokio::test]
async fn nodes_persist_and_list_children() {
    let repo = repo();
    let parent = MemoryNode::new(
        "memory://sessions".to_string(),
        NodeKind::Session,
        None,
        "sessions".to_string(),
        "imported sessions".to_string(),
        String::new(),
        1,
        1,
    );
    let child = MemoryNode::new(
        "memory://sessions/abc".to_string(),
        NodeKind::Session,
        Some("memory://sessions".to_string()),
        "one session".to_string(),
        "a conversation".to_string(),
        "full transcript".to_string(),
        2,
        2,
    );
    repo.upsert_node(&parent, None).await.unwrap();
    repo.upsert_node(&child, Some(&[1.0, 0.0, 0.0, 0.0]))
        .await
        .unwrap();

    let found = repo
        .find_node("memory://sessions/abc")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(found.content(), "full transcript");

    let children = repo.list_child_nodes("memory://sessions").await.unwrap();
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].uri(), "memory://sessions/abc");

    let semantic = repo
        .search_nodes_semantic(&[1.0, 0.0, 0.0, 0.0], None, 10)
        .await
        .unwrap();
    assert_eq!(semantic[0].0.uri(), "memory://sessions/abc");

    assert!(repo.delete_node("memory://sessions/abc").await.unwrap());
    assert!(repo
        .find_node("memory://sessions/abc")
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn sessions_and_dream_runs_persist() {
    let repo = repo();
    let session = ImportedSession {
        id: "sess-1".to_string(),
        source: "claude".to_string(),
        imported_at: 100,
        message_count: 12,
        items_written: 3,
        status: SessionStatus::Imported,
        last_error: None,
    };
    repo.record_session(&session).await.unwrap();
    let got = repo.find_session("sess-1").await.unwrap().unwrap();
    assert_eq!(got.items_written, 3);
    assert_eq!(repo.list_sessions().await.unwrap().len(), 1);

    let run = DreamRun {
        id: "dream-1".to_string(),
        started_at: 1,
        finished_at: 2,
        sessions_imported: 1,
        clusters_found: 0,
        operations_applied: 0,
        operations_skipped: 0,
        status: "completed".to_string(),
    };
    repo.record_dream_run(&run).await.unwrap();
    let last = repo.last_dream_run().await.unwrap().unwrap();
    assert_eq!(last.id, "dream-1");
}

#[tokio::test]
async fn stats_count_items_sessions_and_nodes() {
    let repo = repo();
    repo.upsert_item(&item("a", "one", None), None)
        .await
        .unwrap();
    repo.upsert_item(&item("b", "two", None), None)
        .await
        .unwrap();
    repo.record_session(&ImportedSession {
        id: "s".to_string(),
        source: "claude".to_string(),
        imported_at: 1,
        message_count: 1,
        items_written: 2,
        status: SessionStatus::Imported,
        last_error: None,
    })
    .await
    .unwrap();

    let stats = repo.stats().await.unwrap();
    assert_eq!(stats.total_items, 2);
    assert_eq!(stats.total_sessions, 1);
}

#[tokio::test]
async fn wrong_dimension_vector_is_rejected() {
    let repo = repo();
    let it = item("bad", "wrong-width vector", None);
    let err = repo.upsert_item(&it, Some(&[1.0, 0.0])).await;
    assert!(
        err.is_err(),
        "a {DIMS}-dim store must reject a 2-dim vector"
    );
}

#[tokio::test]
async fn stored_embedding_model_wins_on_reopen() {
    // The embedding model that wrote the vectors is authoritative for
    // retrieval. Reopening a store with a *different* configured model must
    // adopt the stored one (not reject, not overwrite), so query embeddings
    // stay comparable with the stored vectors.
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("memory.duckdb");

    let first = DuckdbMemoryRepository::new(&db_path, DIMS, "model-A").unwrap();
    assert_eq!(first.stored_embedding_model(), "model-A");
    assert_eq!(first.embedding_model().await.unwrap(), "model-A");
    drop(first);

    // Reopen with a different configured model — the store keeps model-A.
    let reopened = DuckdbMemoryRepository::new(&db_path, DIMS, "model-B").unwrap();
    assert_eq!(
        reopened.stored_embedding_model(),
        "model-A",
        "the stored model must win over the newly-configured one"
    );
    assert_eq!(reopened.embedding_model().await.unwrap(), "model-A");
}

#[tokio::test]
async fn dimension_mismatch_on_reopen_is_still_rejected() {
    // Dimensions remain a hard pin: different-width vectors are incomparable,
    // so reopening at a different dimension is a genuine error (unlike the
    // model, which is adopted).
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("memory.duckdb");

    DuckdbMemoryRepository::new(&db_path, 4, "model-A").unwrap();
    let err = DuckdbMemoryRepository::new(&db_path, 8, "model-A");
    assert!(
        err.is_err(),
        "reopening a 4-dim store at 8 dims must be rejected"
    );
}

#[tokio::test]
async fn namespace_crud_and_membership() {
    let repo = repo();

    // Create is idempotent.
    assert!(repo.create_namespace("payments").await.unwrap());
    assert!(!repo.create_namespace("payments").await.unwrap());

    // An empty namespace exists but lists no projects.
    assert_eq!(repo.namespace_projects("payments").await.unwrap().len(), 0);
    assert_eq!(
        repo.list_namespaces().await.unwrap(),
        vec![("payments".to_string(), 0)]
    );

    // Assign is idempotent and reflected in the count + projects.
    assert!(repo
        .assign_project("payments", "svc-billing")
        .await
        .unwrap());
    assert!(!repo
        .assign_project("payments", "svc-billing")
        .await
        .unwrap());
    assert!(repo.assign_project("payments", "svc-ledger").await.unwrap());
    assert_eq!(
        repo.namespace_projects("payments").await.unwrap(),
        vec!["svc-billing".to_string(), "svc-ledger".to_string()]
    );
    assert_eq!(
        repo.list_namespaces().await.unwrap(),
        vec![("payments".to_string(), 2)]
    );

    // Unassign removes membership; the namespace persists.
    assert!(repo
        .unassign_project("payments", "svc-ledger")
        .await
        .unwrap());
    assert!(!repo
        .unassign_project("payments", "svc-ledger")
        .await
        .unwrap());
    assert_eq!(
        repo.namespace_projects("payments").await.unwrap(),
        vec!["svc-billing".to_string()]
    );

    // Delete removes the namespace entirely.
    assert!(repo.delete_namespace("payments").await.unwrap());
    assert!(!repo.delete_namespace("payments").await.unwrap());
    assert!(repo.list_namespaces().await.unwrap().is_empty());
}

#[tokio::test]
async fn assigning_a_project_autocreates_the_namespace() {
    let repo = repo();
    // Assigning to a not-yet-created namespace creates it (with the member).
    assert!(repo.assign_project("infra", "svc-a").await.unwrap());
    assert_eq!(
        repo.list_namespaces().await.unwrap(),
        vec![("infra".to_string(), 1)]
    );
}

#[tokio::test]
async fn namespace_scoped_search_returns_globals_plus_member_projects() {
    let repo = repo();
    // A global fact, two project facts (in the namespace), and one outside it.
    repo.upsert_item(&item("global_fact", "shared knowledge", None), None)
        .await
        .unwrap();
    repo.upsert_item(&item("a_fact", "billing detail", Some("svc-billing")), None)
        .await
        .unwrap();
    repo.upsert_item(&item("b_fact", "ledger detail", Some("svc-ledger")), None)
        .await
        .unwrap();
    repo.upsert_item(
        &item("other_fact", "unrelated repo", Some("svc-other")),
        None,
    )
    .await
    .unwrap();

    // Scope = the namespace's two member projects.
    let scope = vec!["svc-billing".to_string(), "svc-ledger".to_string()];
    let hits = repo
        .search_keyword(
            "detail knowledge billing ledger unrelated",
            None,
            Some(&scope),
            50,
        )
        .await
        .unwrap();
    let names: std::collections::HashSet<&str> = hits.iter().map(|(i, _)| i.name()).collect();

    assert!(names.contains("global_fact"), "globals are always in scope");
    assert!(
        names.contains("a_fact"),
        "member project svc-billing in scope"
    );
    assert!(
        names.contains("b_fact"),
        "member project svc-ledger in scope"
    );
    assert!(
        !names.contains("other_fact"),
        "a project outside the namespace must be excluded"
    );
}

#[tokio::test]
async fn empty_scope_restricts_to_globals_only() {
    let repo = repo();
    repo.upsert_item(&item("g", "global", None), None)
        .await
        .unwrap();
    repo.upsert_item(&item("p", "project scoped", Some("svc-a")), None)
        .await
        .unwrap();

    // An empty scope slice means "globals only".
    let hits = repo
        .search_keyword("global project scoped", None, Some(&[]), 50)
        .await
        .unwrap();
    let names: std::collections::HashSet<&str> = hits.iter().map(|(i, _)| i.name()).collect();
    assert!(names.contains("g"));
    assert!(!names.contains("p"), "empty scope excludes project items");
}

#[tokio::test]
async fn empty_namespace_name_is_rejected() {
    let repo = repo();
    assert!(repo.create_namespace("   ").await.is_err());
    assert!(repo.assign_project("ns", "").await.is_err());
}
