//! Storage-semantics tests for the DuckDB memory-graph adapter.
//!
//! These drive the real [`MemoryRepository`] implementation on
//! [`DuckdbStore`] — deliberately, rather than a fake. Every defect
//! worth catching in this adapter lives in SQL the port signature cannot
//! express: the `FLOAT[dimensions]` literal width, `LIKE` metacharacter
//! escaping, the `status = 'active'` recall filter, the multi-project `IN` list,
//! and the positional row indices `memory_from_row` reads *by number*. A fake
//! satisfies the trait and proves none of that. An in-memory database and small
//! hand-written vectors keep every test deterministic and fast.

use std::sync::Arc;

use memory_rs::Predicate;
use memory_rs::{
    DuckdbStore, EdgeOrigin, EdgeType, Entity, EntityRef, ImportedSession, Memory, MemoryEdge,
    MemoryKind, MemoryNode, MemoryRepository, MemoryStatus, NodeKind, NodeRepository,
    SessionStatus, SourceKind,
};

const DIMS: usize = 4;

fn repo() -> Arc<dyn MemoryRepository> {
    Arc::new(DuckdbStore::in_memory(DIMS, "mock-embedding").unwrap())
}

/// A minimal `active` memory. Tests overwrite only the fields they exercise, so
/// the assertion reads as the difference from a known baseline.
fn memory(id: &str, statement: &str) -> Memory {
    Memory {
        id: id.to_string(),
        kind: MemoryKind::Fact,
        subject: EntityRef::Entity("entity-team".to_string()),
        predicate: Predicate::Uses,
        object: EntityRef::Literal("tabs".to_string()),
        statement: statement.to_string(),
        project: None,
        recorded_at: 1_700_000_000,
        valid_from: 1_700_000_000,
        valid_to: None,
        source_session_id: Some("session-1".to_string()),
        source_message_index: Some(3),
        source_kind: SourceKind::AssistantInferred,
        confidence: 0.8,
        status: MemoryStatus::Active,
        derived: false,
        derived_from: Vec::new(),
    }
}

fn entity(id: &str, canonical_name: &str, names: &[&str]) -> Entity {
    Entity {
        id: id.to_string(),
        entity_type: "project".to_string(),
        canonical_name: canonical_name.to_string(),
        names: names.iter().map(|a| a.to_string()).collect(),
        created_at: 10,
        updated_at: 20,
    }
}

fn edge(from: &str, to: &str, edge_type: EdgeType, created_at: i64, confidence: f32) -> MemoryEdge {
    MemoryEdge {
        from_memory: from.to_string(),
        to_memory: to.to_string(),
        edge_type,
        created_at,
        created_by: EdgeOrigin::Ingestion,
        confidence,
    }
}

fn ids(memories: &[Memory]) -> Vec<&str> {
    memories.iter().map(|c| c.id.as_str()).collect()
}

fn hit_ids(hits: &[(Memory, f32)]) -> Vec<&str> {
    hits.iter().map(|(c, _)| c.id.as_str()).collect()
}

fn projects(names: &[&str]) -> Vec<String> {
    names.iter().map(|p| p.to_string()).collect()
}

// ── Round-trip and identity ──────────────────────────────────────────────

#[tokio::test]
async fn memory_round_trips_with_every_field_preserved() {
    let repo = repo();

    // Literal subject, entity object, project scope, closed validity window,
    // derived with provenance — every optional field populated.
    let mut rich = memory("memory-rich", "the team moved off the legacy runner");
    rich.kind = MemoryKind::Experience;
    rich.subject = EntityRef::Literal("the team".to_string());
    rich.object = EntityRef::Entity("entity-runner".to_string());
    rich.project = Some("svc-a".to_string());
    rich.valid_from = 1_699_000_000;
    rich.valid_to = Some(1_700_000_900);
    rich.source_session_id = Some("session-2".to_string());
    rich.source_message_index = Some(7);
    rich.source_kind = SourceKind::Derived;
    rich.confidence = 0.25;
    rich.status = MemoryStatus::Superseded;
    rich.derived = true;
    rich.derived_from = vec!["memory-a".to_string(), "memory-b".to_string()];
    repo.append_memory(&rich, Some(&[0.5, 0.5, 0.5, 0.5]))
        .await
        .unwrap();
    assert_eq!(
        repo.find_memory("memory-rich").await.unwrap().unwrap(),
        rich
    );

    // The mirror image: entity subject, literal object, global scope, open
    // validity window, no provenance, no vector. `project: None` must survive
    // the empty-string flattening rather than coming back as `Some("")`.
    let mut sparse = memory("memory-sparse", "indentation is tabs");
    sparse.subject = EntityRef::Entity("entity-team".to_string());
    sparse.object = EntityRef::Literal("tabs".to_string());
    sparse.project = None;
    sparse.valid_to = None;
    sparse.source_session_id = None;
    sparse.source_message_index = None;
    sparse.source_kind = SourceKind::UserStated;
    repo.append_memory(&sparse, None).await.unwrap();
    assert_eq!(
        repo.find_memory("memory-sparse").await.unwrap().unwrap(),
        sparse
    );

    assert!(repo.find_memory("memory-missing").await.unwrap().is_none());
}

#[tokio::test]
async fn re_appending_an_existing_id_replaces_the_row_and_its_vector() {
    let repo = repo();
    repo.append_memory(
        &memory("memory-1", "the runner is nightly"),
        Some(&[1.0, 0.0, 0.0, 0.0]),
    )
    .await
    .unwrap();

    let mut second = memory("memory-1", "the runner is hourly");
    second.confidence = 0.42;
    repo.append_memory(&second, Some(&[0.0, 1.0, 0.0, 0.0]))
        .await
        .unwrap();

    let all = repo.list_memories(None, None, None).await.unwrap();
    assert_eq!(
        all.len(),
        1,
        "re-appending an id must not duplicate the row"
    );
    assert_eq!(all[0].statement, "the runner is hourly");

    let vectors = repo.list_memory_embeddings().await.unwrap();
    assert_eq!(vectors.len(), 1, "the replaced vector must not be orphaned");
    assert_eq!(vectors[0].0, "memory-1");
    assert_eq!(vectors[0].1, vec![0.0, 1.0, 0.0, 0.0]);

    // Re-appending without an embedding drops the vector instead of keeping the
    // stale one. The memory stays keyword-searchable but leaves semantic recall,
    // which is the honest outcome for a statement nothing has embedded — a
    // retained vector would answer queries about the *previous* statement.
    repo.append_memory(&memory("memory-1", "the runner is on demand"), None)
        .await
        .unwrap();
    assert!(repo.list_memory_embeddings().await.unwrap().is_empty());
}

#[tokio::test]
async fn find_memories_batches_known_ids_and_skips_unknown_ones() {
    let repo = repo();
    for id in ["memory-1", "memory-2", "memory-3"] {
        repo.append_memory(&memory(id, "a statement"), None)
            .await
            .unwrap();
    }

    let mut found = repo
        .find_memories(&[
            "memory-1".to_string(),
            "memory-missing".to_string(),
            "memory-3".to_string(),
        ])
        .await
        .unwrap();
    found.sort_by(|a, b| a.id.cmp(&b.id));
    assert_eq!(
        ids(&found),
        vec!["memory-1", "memory-3"],
        "unknown ids are skipped, not an error"
    );

    assert!(repo.find_memories(&[]).await.unwrap().is_empty());
}

// ── Filtering ────────────────────────────────────────────────────────────

/// Seed a small graph spanning three projects, two kinds and two statuses.
async fn seed_scoped_memories(repo: &Arc<dyn MemoryRepository>) {
    let rows: [(&str, MemoryKind, MemoryStatus, Option<&str>, i64); 5] = [
        (
            "memory-global",
            MemoryKind::Fact,
            MemoryStatus::Active,
            None,
            1,
        ),
        (
            "memory-a-pref",
            MemoryKind::Preference,
            MemoryStatus::Active,
            Some("svc-a"),
            2,
        ),
        (
            "memory-a-old",
            MemoryKind::Fact,
            MemoryStatus::Superseded,
            Some("svc-a"),
            3,
        ),
        (
            "memory-b",
            MemoryKind::Fact,
            MemoryStatus::Active,
            Some("svc-b"),
            4,
        ),
        (
            "memory-c",
            MemoryKind::Fact,
            MemoryStatus::Active,
            Some("svc-c"),
            5,
        ),
    ];
    for (id, kind, status, project, recorded_at) in rows {
        let mut c = memory(id, "the pipeline publishes an artifact");
        c.kind = kind;
        c.status = status;
        c.project = project.map(str::to_string);
        c.recorded_at = recorded_at;
        repo.append_memory(&c, None).await.unwrap();
    }
}

#[tokio::test]
async fn list_memories_filters_by_kind_status_and_scope_independently_and_together() {
    let repo = repo();
    seed_scoped_memories(&repo).await;

    // Newest first, so a caller paging the log sees recent memories without
    // sorting client-side.
    let all = repo.list_memories(None, None, None).await.unwrap();
    assert_eq!(
        ids(&all),
        vec![
            "memory-c",
            "memory-b",
            "memory-a-old",
            "memory-a-pref",
            "memory-global"
        ],
    );

    let facts = repo
        .list_memories(Some(MemoryKind::Fact), None, None)
        .await
        .unwrap();
    assert_eq!(
        ids(&facts),
        vec!["memory-c", "memory-b", "memory-a-old", "memory-global"]
    );

    let active = repo
        .list_memories(None, Some(MemoryStatus::Active), None)
        .await
        .unwrap();
    assert_eq!(
        ids(&active),
        vec!["memory-c", "memory-b", "memory-a-pref", "memory-global"]
    );

    let scoped = repo
        .list_memories(None, None, Some(&projects(&["svc-a"])))
        .await
        .unwrap();
    assert_eq!(
        ids(&scoped),
        vec!["memory-a-old", "memory-a-pref", "memory-global"]
    );

    // All three at once — the combination is where a mis-joined clause shows up.
    let combined = repo
        .list_memories(
            Some(MemoryKind::Fact),
            Some(MemoryStatus::Active),
            Some(&projects(&["svc-a", "svc-b"])),
        )
        .await
        .unwrap();
    assert_eq!(ids(&combined), vec!["memory-b", "memory-global"]);
}

/// A namespace resolves to *many* projects, which is why the port takes a slice
/// rather than a single `Option<&str>`. Both members plus globals are in scope;
/// a project outside the namespace is not.
#[tokio::test]
async fn list_memories_namespace_scope_spans_every_member_project_plus_globals() {
    let repo = repo();
    seed_scoped_memories(&repo).await;

    let namespace = projects(&["svc-a", "svc-b"]);
    let hits = repo
        .list_memories(None, None, Some(&namespace))
        .await
        .unwrap();
    assert_eq!(
        ids(&hits),
        vec!["memory-b", "memory-a-old", "memory-a-pref", "memory-global"],
        "both member projects and globals are in scope, svc-c is not"
    );
}

#[tokio::test]
async fn list_memories_empty_scope_is_globals_only_and_no_scope_is_everything() {
    let repo = repo();
    seed_scoped_memories(&repo).await;

    let globals = repo.list_memories(None, None, Some(&[])).await.unwrap();
    assert_eq!(
        ids(&globals),
        vec!["memory-global"],
        "an empty scope slice means globals only"
    );

    assert_eq!(repo.list_memories(None, None, None).await.unwrap().len(), 5);
}

// ── The exclusion invariant ──────────────────────────────────────────────

/// Only `active` memories may surface from recall — and that includes
/// superseded links in a chain, and retractions.
/// Excluding them here rather than downstream is what lets every caller stop
/// worrying about filtering an unresolved contradiction out of its results.
#[tokio::test]
async fn non_active_memories_are_excluded_from_both_search_legs() {
    for retired in [MemoryStatus::Superseded, MemoryStatus::Retracted] {
        let repo = repo();
        let vector = [1.0, 0.0, 0.0, 0.0];

        repo.append_memory(
            &memory("memory-active", "the deploy job runs on merge"),
            Some(&vector),
        )
        .await
        .unwrap();

        // Identical statement and vector, so only the status can separate them.
        let mut hidden = memory("memory-hidden", "the deploy job runs on merge");
        hidden.status = retired;
        repo.append_memory(&hidden, Some(&vector)).await.unwrap();

        let semantic = repo
            .search_memories_semantic(&vector, None, None, 10)
            .await
            .unwrap();
        assert_eq!(
            hit_ids(&semantic),
            vec!["memory-active"],
            "{} must not surface from semantic recall",
            retired.as_str(),
        );

        let keyword = repo
            .search_memories_keyword("deploy job", None, None, 10)
            .await
            .unwrap();
        assert_eq!(
            hit_ids(&keyword),
            vec!["memory-active"],
            "{} must not surface from keyword recall",
            retired.as_str(),
        );
    }
}

// ── Lifecycle ────────────────────────────────────────────────────────────

#[tokio::test]
async fn set_memory_status_without_a_timestamp_leaves_valid_to_intact() {
    let repo = repo();
    let mut c = memory("memory-1", "the runner is nightly");
    c.valid_to = Some(1_700_000_500);
    repo.append_memory(&c, None).await.unwrap();

    // Retracting must not silently rewrite when the memory stopped being true.
    assert!(repo
        .set_memory_status("memory-1", MemoryStatus::Retracted, None)
        .await
        .unwrap());
    let after = repo.find_memory("memory-1").await.unwrap().unwrap();
    assert_eq!(after.status, MemoryStatus::Retracted);
    assert_eq!(after.valid_to, Some(1_700_000_500));

    // With a timestamp, both move.
    assert!(repo
        .set_memory_status("memory-1", MemoryStatus::Superseded, Some(1_700_009_999))
        .await
        .unwrap());
    let after = repo.find_memory("memory-1").await.unwrap().unwrap();
    assert_eq!(after.status, MemoryStatus::Superseded);
    assert_eq!(after.valid_to, Some(1_700_009_999));
}

/// The one thing `set_memory_status` refuses to do: a reopened memory carrying a
/// stale `valid_to` would assert it stopped being true at a moment it did not.
#[tokio::test]
async fn reopen_memory_restores_active_and_clears_valid_to() {
    let repo = repo();
    let mut c = memory("memory-1", "the runner is nightly");
    c.status = MemoryStatus::Superseded;
    c.valid_to = Some(1_700_000_500);
    repo.append_memory(&c, None).await.unwrap();

    assert!(repo.reopen_memory("memory-1").await.unwrap());
    let after = repo.find_memory("memory-1").await.unwrap().unwrap();
    assert_eq!(after.status, MemoryStatus::Active);
    assert_eq!(after.valid_to, None);
}

#[tokio::test]
async fn lifecycle_transitions_report_an_unknown_memory() {
    let repo = repo();
    assert!(!repo
        .set_memory_status("memory-missing", MemoryStatus::Retracted, Some(1))
        .await
        .unwrap());
    assert!(!repo
        .set_memory_status("memory-missing", MemoryStatus::Retracted, None)
        .await
        .unwrap());
    assert!(!repo.reopen_memory("memory-missing").await.unwrap());
}

#[tokio::test]
async fn deleting_a_session_removes_its_memories_vectors_and_edges() {
    let repo = repo();
    for id in ["memory-1", "memory-2"] {
        let mut c = memory(id, "a statement from the session");
        c.source_session_id = Some("session-1".to_string());
        repo.append_memory(&c, Some(&[1.0, 0.0, 0.0, 0.0]))
            .await
            .unwrap();
    }
    let mut keeper = memory("memory-3", "a statement from elsewhere");
    keeper.source_session_id = Some("session-2".to_string());
    repo.append_memory(&keeper, Some(&[0.0, 1.0, 0.0, 0.0]))
        .await
        .unwrap();
    repo.add_edge(&edge("memory-2", "memory-1", EdgeType::Supersedes, 1, 1.0))
        .await
        .unwrap();

    assert_eq!(
        repo.count_memories_for_session("session-1").await.unwrap(),
        2
    );
    assert_eq!(
        repo.delete_memories_for_session("session-1").await.unwrap(),
        2
    );
    assert_eq!(
        repo.count_memories_for_session("session-1").await.unwrap(),
        0
    );

    assert_eq!(
        ids(&repo.list_memories(None, None, None).await.unwrap()),
        vec!["memory-3"]
    );
    assert_eq!(
        repo.list_memory_embeddings().await.unwrap().len(),
        1,
        "the deleted memories' vectors go with them"
    );
    assert!(
        repo.edges_from("memory-2").await.unwrap().is_empty(),
        "no edge may outlive the memory it points at"
    );
    assert_eq!(repo.memory_stats().await.unwrap().total_edges, 0);
}

// ── Edges ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn add_edge_is_keyed_on_from_to_and_type() {
    let repo = repo();
    repo.add_edge(&edge("memory-2", "memory-1", EdgeType::Supersedes, 1, 0.5))
        .await
        .unwrap();
    repo.add_edge(&edge("memory-2", "memory-1", EdgeType::Supersedes, 2, 0.9))
        .await
        .unwrap();

    let edges = repo.edges_from("memory-2").await.unwrap();
    assert_eq!(
        edges.len(),
        1,
        "the same triple replaces rather than duplicates"
    );
    assert_eq!(edges[0].confidence, 0.9);
    assert_eq!(edges[0].created_at, 2);

    // The same pair under a different type is a different edge, not a rewrite.
    repo.add_edge(&edge("memory-2", "memory-1", EdgeType::Contradicts, 3, 0.7))
        .await
        .unwrap();
    let edges = repo.edges_from("memory-2").await.unwrap();
    assert_eq!(edges.len(), 2);
    assert_eq!(edges[1].edge_type, EdgeType::Contradicts);
}

#[tokio::test]
async fn edges_from_and_edges_to_do_not_leak_the_other_direction() {
    let repo = repo();
    let e = edge("memory-2", "memory-1", EdgeType::Supersedes, 1, 0.5);
    repo.add_edge(&e).await.unwrap();

    assert_eq!(repo.edges_from("memory-2").await.unwrap(), vec![e.clone()]);
    assert!(repo.edges_from("memory-1").await.unwrap().is_empty());
    assert_eq!(repo.edges_to("memory-1").await.unwrap(), vec![e]);
    assert!(repo.edges_to("memory-2").await.unwrap().is_empty());
}

// ── Entities ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn entity_round_trips_with_canonical_name_and_folded_names() {
    let repo = repo();
    // "Payments" only differs from "payments" by case, "  payments-svc  " by
    // whitespace, and the canonical name must itself resolve as a name.
    let e = entity(
        "entity-1",
        "Payments Service",
        &["payments", "Payments", "  payments-svc  ", ""],
    );
    repo.upsert_entity(&e, None).await.unwrap();

    let mut want = e.clone();
    // Names fold on the *normalized* key, so "payments" is not stored beside
    // "Payments Service" — they are the same key, and the whole point of the
    // key is that either spelling resolves. The first form written wins the
    // display row; the rest would be noise in a name list.
    want.names = vec!["Payments Service".to_string(), "payments-svc".to_string()];
    assert_eq!(repo.find_entity("entity-1").await.unwrap().unwrap(), want);
    assert_eq!(repo.list_entities().await.unwrap(), vec![want]);

    assert!(repo.find_entity("entity-missing").await.unwrap().is_none());
}

#[tokio::test]
async fn find_entities_by_name_matches_case_and_role_word_variants() {
    let repo = repo();
    repo.upsert_entity(
        &entity("entity-1", "Payments Service", &["payments-svc"]),
        None,
    )
    .await
    .unwrap();

    // The last three are the point: the lookup key is normalized, so a name
    // nobody wrote down still lands on the entity instead of minting a second.
    for name in [
        "PAYMENTS-SVC",
        "payments-svc",
        "payments service",
        "payments",
        "the payments package",
        "Payments",
    ] {
        let found = repo.find_entities_by_name(name).await.unwrap();
        assert_eq!(
            found.first().map(|e| e.id.as_str()),
            Some("entity-1"),
            "name should resolve: {name}",
        );
    }
    assert!(repo
        .find_entities_by_name("ledger")
        .await
        .unwrap()
        .is_empty());
}

/// One normalized key can front two entities — "foo" the tool and "foo" the
/// project are not the same thing. The lookup returns both and lets the caller
/// apply the type guard; returning only the first would hand back the wrong one
/// half the time.
#[tokio::test]
async fn find_entities_by_name_returns_every_entity_sharing_a_key() {
    let repo = repo();
    let mut tool = entity("entity-tool", "octo", &[]);
    tool.entity_type = "tool".to_string();
    let mut project = entity("entity-project", "the octo repository", &[]);
    project.entity_type = "project".to_string();
    repo.upsert_entity(&tool, None).await.unwrap();
    repo.upsert_entity(&project, None).await.unwrap();

    let mut ids: Vec<String> = repo
        .find_entities_by_name("octo")
        .await
        .unwrap()
        .into_iter()
        .map(|e| e.id)
        .collect();
    ids.sort();
    assert_eq!(ids, ["entity-project", "entity-tool"]);
}

/// A merge has to move *both* reference columns. One entity is the subject of
/// one memory and the object of another here on purpose: an implementation that
/// updated only `subject_entity_id` would strand half the graph on a dead id,
/// and a test that only inspected subjects would not notice.
#[tokio::test]
async fn repoint_entity_moves_subject_and_object_references() {
    let repo = repo();

    let mut as_subject = memory("memory-subject", "the old entity owns billing");
    as_subject.subject = EntityRef::Entity("entity-old".to_string());
    as_subject.object = EntityRef::Literal("billing".to_string());
    repo.append_memory(&as_subject, None).await.unwrap();

    let mut as_object = memory("memory-object", "billing is owned by the old entity");
    as_object.subject = EntityRef::Entity("entity-other".to_string());
    as_object.object = EntityRef::Entity("entity-old".to_string());
    repo.append_memory(&as_object, None).await.unwrap();

    let mut untouched = memory("memory-untouched", "something unrelated");
    untouched.subject = EntityRef::Entity("entity-other".to_string());
    repo.append_memory(&untouched, None).await.unwrap();

    let moved = repo
        .repoint_entity("entity-old", "entity-new")
        .await
        .unwrap();
    assert_eq!(moved, 2, "subject and object references both count");

    let subject_memory = repo.find_memory("memory-subject").await.unwrap().unwrap();
    assert_eq!(
        subject_memory.subject,
        EntityRef::Entity("entity-new".to_string())
    );
    let object_memory = repo.find_memory("memory-object").await.unwrap().unwrap();
    assert_eq!(
        object_memory.object,
        EntityRef::Entity("entity-new".to_string())
    );
    assert_eq!(
        object_memory.subject,
        EntityRef::Entity("entity-other".to_string()),
        "an unrelated reference on the same row must not move"
    );
    assert_eq!(
        repo.find_memory("memory-untouched")
            .await
            .unwrap()
            .unwrap()
            .subject,
        EntityRef::Entity("entity-other".to_string())
    );

    assert_eq!(
        repo.repoint_entity("entity-old", "entity-new")
            .await
            .unwrap(),
        0
    );
}

#[tokio::test]
async fn delete_entity_removes_its_names_and_vector() {
    let repo = repo();
    repo.upsert_entity(
        &entity("entity-1", "Payments Service", &["payments-svc"]),
        Some(&[1.0, 0.0, 0.0, 0.0]),
    )
    .await
    .unwrap();

    assert!(repo.delete_entity("entity-1").await.unwrap());
    assert!(repo.find_entity("entity-1").await.unwrap().is_none());
    assert!(repo
        .find_entities_by_name("payments-svc")
        .await
        .unwrap()
        .is_empty());
    assert!(
        repo.search_entities_semantic(&[1.0, 0.0, 0.0, 0.0], 10)
            .await
            .unwrap()
            .is_empty(),
        "the vector must be deleted with the entity, not orphaned"
    );
    // Deleting again is a no-op.
    assert!(!repo.delete_entity("entity-1").await.unwrap());
}

#[tokio::test]
async fn entity_semantic_search_ranks_by_cosine_distance_and_carries_names() {
    let repo = repo();
    repo.upsert_entity(
        &entity("entity-near", "Payments Service", &["payments-svc"]),
        Some(&[0.9, 0.1, 0.0, 0.0]),
    )
    .await
    .unwrap();
    repo.upsert_entity(
        &entity("entity-far", "Ledger Service", &["ledger-svc"]),
        Some(&[0.0, 1.0, 0.0, 0.0]),
    )
    .await
    .unwrap();

    let hits = repo
        .search_entities_semantic(&[1.0, 0.0, 0.0, 0.0], 10)
        .await
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].0.id, "entity-near", "closest vector ranks first");
    assert!(hits[0].1 > hits[1].1, "{} !> {}", hits[0].1, hits[1].1);
    assert_eq!(
        hits[0].0.names,
        vec!["Payments Service".to_string(), "payments-svc".to_string()],
        "search hits are hydrated with names, like a direct fetch"
    );
}

// ── Search mechanics ─────────────────────────────────────────────────────

#[tokio::test]
async fn memory_semantic_search_ranks_by_cosine_distance_with_unit_scores() {
    let repo = repo();
    repo.append_memory(
        &memory("memory-near", "aligned with the query"),
        Some(&[0.9, 0.1, 0.0, 0.0]),
    )
    .await
    .unwrap();
    repo.append_memory(
        &memory("memory-far", "pointing the other way"),
        Some(&[0.0, 1.0, 0.0, 0.0]),
    )
    .await
    .unwrap();
    // A memory without a vector is keyword-searchable but not semantically.
    repo.append_memory(&memory("memory-unembedded", "no vector at all"), None)
        .await
        .unwrap();

    let hits = repo
        .search_memories_semantic(&[1.0, 0.0, 0.0, 0.0], None, None, 10)
        .await
        .unwrap();
    assert_eq!(hit_ids(&hits), vec!["memory-near", "memory-far"]);
    assert!(hits[0].1 > hits[1].1, "{} !> {}", hits[0].1, hits[1].1);
    for (c, score) in &hits {
        assert!(
            (-1e-6..=1.0 + 1e-6).contains(score),
            "score for {} out of [0, 1]: {score}",
            c.id,
        );
    }
}

#[tokio::test]
async fn keyword_search_scores_the_fraction_of_terms_matched_ignoring_case() {
    let repo = repo();
    repo.append_memory(
        &memory("memory-both", "the Release train ships on Friday"),
        None,
    )
    .await
    .unwrap();
    repo.append_memory(
        &memory("memory-one", "the release notes are generated"),
        None,
    )
    .await
    .unwrap();
    repo.append_memory(&memory("memory-none", "something else entirely"), None)
        .await
        .unwrap();

    let hits = repo
        .search_memories_keyword("RELEASE friday", None, None, 10)
        .await
        .unwrap();
    assert_eq!(
        hit_ids(&hits),
        vec!["memory-both", "memory-one"],
        "matching is case-insensitive in both directions, and non-matches drop out"
    );
    assert!(
        (hits[0].1 - 1.0).abs() < 1e-6,
        "2 of 2 terms: {}",
        hits[0].1
    );
    assert!(
        (hits[1].1 - 0.5).abs() < 1e-6,
        "1 of 2 terms: {}",
        hits[1].1
    );

    // A query with no usable terms is not a match-everything query.
    assert!(repo
        .search_memories_keyword("   ", None, None, 10)
        .await
        .unwrap()
        .is_empty());
}

/// `%` and `_` are `LIKE` metacharacters. Unescaped, a user searching for `50%`
/// would match `501` and a search for `snake_case` would match `snakeXcase` —
/// silent false positives that look like ordinary fuzzy recall.
#[tokio::test]
async fn keyword_search_treats_like_metacharacters_literally() {
    let repo = repo();
    repo.append_memory(
        &memory("memory-percent", "coverage rose 50% last week"),
        None,
    )
    .await
    .unwrap();
    repo.append_memory(&memory("memory-digits", "coverage rose 501 points"), None)
        .await
        .unwrap();
    repo.append_memory(
        &memory("memory-underscore", "identifiers use snake_case"),
        None,
    )
    .await
    .unwrap();
    repo.append_memory(
        &memory("memory-wildcard", "identifiers use snakeXcase"),
        None,
    )
    .await
    .unwrap();

    let percent = repo
        .search_memories_keyword("50%", None, None, 10)
        .await
        .unwrap();
    assert_eq!(
        hit_ids(&percent),
        vec!["memory-percent"],
        "'%' must match a literal percent sign, not any suffix"
    );

    let underscore = repo
        .search_memories_keyword("snake_case", None, None, 10)
        .await
        .unwrap();
    assert_eq!(
        hit_ids(&underscore),
        vec!["memory-underscore"],
        "'_' must match a literal underscore, not any single character"
    );
}

#[tokio::test]
async fn kind_filter_applies_to_both_search_legs() {
    let repo = repo();
    let vector = [1.0, 0.0, 0.0, 0.0];
    let mut fact = memory("memory-fact", "the pipeline publishes an artifact");
    fact.kind = MemoryKind::Fact;
    repo.append_memory(&fact, Some(&vector)).await.unwrap();
    let mut preference = memory("memory-pref", "the pipeline publishes an artifact");
    preference.kind = MemoryKind::Preference;
    repo.append_memory(&preference, Some(&vector))
        .await
        .unwrap();

    let semantic = repo
        .search_memories_semantic(&vector, Some(MemoryKind::Preference), None, 10)
        .await
        .unwrap();
    assert_eq!(hit_ids(&semantic), vec!["memory-pref"]);

    let keyword = repo
        .search_memories_keyword("pipeline artifact", Some(MemoryKind::Preference), None, 10)
        .await
        .unwrap();
    assert_eq!(hit_ids(&keyword), vec!["memory-pref"]);
}

#[tokio::test]
async fn namespace_scope_applies_to_both_search_legs() {
    let repo = repo();
    let vector = [1.0, 0.0, 0.0, 0.0];
    for (id, project) in [
        ("memory-global", None),
        ("memory-a", Some("svc-a")),
        ("memory-b", Some("svc-b")),
        ("memory-c", Some("svc-c")),
    ] {
        let mut c = memory(id, "the pipeline publishes an artifact");
        c.project = project.map(str::to_string);
        repo.append_memory(&c, Some(&vector)).await.unwrap();
    }

    let namespace = projects(&["svc-a", "svc-b"]);
    let semantic_hits = repo
        .search_memories_semantic(&vector, None, Some(&namespace), 10)
        .await
        .unwrap();
    let mut semantic = hit_ids(&semantic_hits);
    semantic.sort_unstable();
    assert_eq!(semantic, vec!["memory-a", "memory-b", "memory-global"]);

    let keyword_hits = repo
        .search_memories_keyword("pipeline artifact", None, Some(&namespace), 10)
        .await
        .unwrap();
    let mut keyword = hit_ids(&keyword_hits);
    keyword.sort_unstable();
    assert_eq!(keyword, vec!["memory-a", "memory-b", "memory-global"]);

    // Globals only.
    let semantic = repo
        .search_memories_semantic(&vector, None, Some(&[]), 10)
        .await
        .unwrap();
    assert_eq!(hit_ids(&semantic), vec!["memory-global"]);
    let keyword = repo
        .search_memories_keyword("pipeline artifact", None, Some(&[]), 10)
        .await
        .unwrap();
    assert_eq!(hit_ids(&keyword), vec!["memory-global"]);
}

#[tokio::test]
async fn wrong_dimension_vectors_are_rejected() {
    let repo = repo();
    assert!(
        repo.append_memory(&memory("memory-1", "bad vector"), Some(&[1.0, 0.0]))
            .await
            .is_err(),
        "a {DIMS}-dim store must reject a 2-dim memory vector"
    );
    assert!(
        repo.upsert_entity(
            &entity("entity-1", "Payments Service", &[]),
            Some(&[1.0, 0.0, 0.0, 0.0, 0.0])
        )
        .await
        .is_err(),
        "a {DIMS}-dim store must reject a 5-dim entity vector"
    );
}

// ── Stats ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn memory_stats_count_the_whole_graph_and_the_breakdowns_add_up() {
    let repo = repo();
    seed_scoped_memories(&repo).await;
    repo.upsert_entity(&entity("entity-1", "Payments Service", &[]), None)
        .await
        .unwrap();
    repo.upsert_entity(&entity("entity-2", "Ledger Service", &[]), None)
        .await
        .unwrap();
    repo.add_edge(&edge(
        "memory-b",
        "memory-a-old",
        EdgeType::Supersedes,
        1,
        1.0,
    ))
    .await
    .unwrap();

    let stats = repo.memory_stats().await.unwrap();
    assert_eq!(stats.total_memories, 5);
    assert_eq!(stats.total_entities, 2);
    assert_eq!(stats.total_edges, 1);

    let by_status: u64 = stats.memories_by_status.iter().map(|(_, n)| n).sum();
    let by_kind: u64 = stats.memories_by_kind.iter().map(|(_, n)| n).sum();
    assert_eq!(
        by_status, stats.total_memories,
        "{:?}",
        stats.memories_by_status
    );
    assert_eq!(
        by_kind, stats.total_memories,
        "{:?}",
        stats.memories_by_kind
    );
    assert_eq!(
        stats.memories_by_status,
        vec![("active".to_string(), 4), ("superseded".to_string(), 1)]
    );
    assert_eq!(
        stats.memories_by_kind,
        vec![("fact".to_string(), 4), ("preference".to_string(), 1)]
    );
}

// ── The shared connection ────────────────────────────────────────────────

/// One repository instance must serve both ports off **one** DuckDB connection.
///
/// The regression this catches: someone "simplifies" the memory adapter into
/// opening its own `Connection`. DuckDB permits a single writer per file, so a
/// second handle to `memory.duckdb` is either a startup lock failure the desktop
/// app has to explain to a user, or — worse — a second database that silently
/// swallows every memory ever written.
///
/// It is deliberately file-backed rather than in-memory: with `:memory:` a
/// stray second connection would just be a second empty database, and both
/// halves of this test would still pass. Reopening the file afterwards is what
/// proves the memory and the node landed in the same place.
#[tokio::test]
async fn memories_and_nodes_share_one_connection() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("memory.duckdb");

    let repo = Arc::new(DuckdbStore::new(&db_path, DIMS, "mock-embedding").unwrap());
    // Exactly how the container hands the store out: one concrete instance
    // behind two port-shaped handles.
    let nodes: Arc<dyn NodeRepository> = repo.clone();
    let memories: Arc<dyn MemoryRepository> = repo.clone();

    let node = MemoryNode::new(
        "memory://sessions/session-1".to_string(),
        NodeKind::Session,
        None,
        "a conversation".to_string(),
        String::new(),
        "full transcript".to_string(),
        1,
        1,
    );
    nodes
        .upsert_node(&node, Some(&[1.0, 0.0, 0.0, 0.0]))
        .await
        .unwrap();
    nodes
        .record_session(&ImportedSession {
            id: "session-1".to_string(),
            source: "claude".to_string(),
            imported_at: 100,
            message_count: 12,
            project: None,
            items_written: 3,
            status: SessionStatus::Imported,
            last_error: None,
        })
        .await
        .unwrap();

    let mut c = memory("memory-1", "the session produced a memory");
    c.source_session_id = Some("session-1".to_string());
    memories
        .append_memory(&c, Some(&[0.0, 1.0, 0.0, 0.0]))
        .await
        .unwrap();

    // Both ports read back from the one live instance.
    assert!(nodes
        .find_node("memory://sessions/session-1")
        .await
        .unwrap()
        .is_some());
    assert!(nodes.find_session("session-1").await.unwrap().is_some());
    assert!(memories.find_memory("memory-1").await.unwrap().is_some());

    drop(nodes);
    drop(memories);
    drop(repo);

    // And both writes are in the same file — a memory adapter with its own
    // connection would have written somewhere else (or failed to open at all).
    let reopened = DuckdbStore::new(&db_path, DIMS, "mock-embedding").unwrap();
    assert!(reopened
        .find_node("memory://sessions/session-1")
        .await
        .unwrap()
        .is_some());
    assert_eq!(
        reopened
            .find_memory("memory-1")
            .await
            .unwrap()
            .unwrap()
            .statement,
        "the session produced a memory"
    );
    assert_eq!(
        reopened
            .count_memories_for_session("session-1")
            .await
            .unwrap(),
        1
    );
}

/// `edges_for` is the batched replacement for per-memory `edges_from` +
/// `edges_to`, so it must return edges in BOTH directions and tolerate ids that
/// have no edges at all.
#[tokio::test]
async fn edges_for_returns_both_directions_in_one_query() {
    let repo = repo();
    for id in ["m-a", "m-b", "m-c", "m-lonely"] {
        repo.append_memory(&memory(id, "statement"), None)
            .await
            .unwrap();
    }
    // b -> a (inbound to a), a -> c (outbound from a)
    for (from, to, kind) in [
        ("m-b", "m-a", EdgeType::Corroborates),
        ("m-a", "m-c", EdgeType::Supersedes),
    ] {
        repo.add_edge(&MemoryEdge {
            from_memory: from.to_string(),
            to_memory: to.to_string(),
            edge_type: kind,
            created_at: 100,
            created_by: EdgeOrigin::Ingestion,
            confidence: 0.9,
        })
        .await
        .unwrap();
    }

    let edges = repo
        .edges_for(&["m-a".to_string(), "m-lonely".to_string()])
        .await
        .unwrap();
    assert_eq!(edges.len(), 2, "expected both directions, got {edges:?}");
    assert!(edges
        .iter()
        .any(|e| e.from_memory == "m-b" && e.to_memory == "m-a"));
    assert!(edges
        .iter()
        .any(|e| e.from_memory == "m-a" && e.to_memory == "m-c"));

    assert!(repo.edges_for(&[]).await.unwrap().is_empty());
}

/// `list_edges` is what makes the conflict queue derived rather than stored.
#[tokio::test]
async fn list_edges_filters_by_type() {
    let repo = repo();
    for id in ["m-a", "m-b"] {
        repo.append_memory(&memory(id, "statement"), None)
            .await
            .unwrap();
    }
    for kind in [EdgeType::Contradicts, EdgeType::Corroborates] {
        repo.add_edge(&MemoryEdge {
            from_memory: "m-a".to_string(),
            to_memory: "m-b".to_string(),
            edge_type: kind,
            created_at: 100,
            created_by: EdgeOrigin::Ingestion,
            confidence: 0.9,
        })
        .await
        .unwrap();
    }
    assert_eq!(repo.list_edges(None).await.unwrap().len(), 2);
    let contradictions = repo.list_edges(Some(EdgeType::Contradicts)).await.unwrap();
    assert_eq!(contradictions.len(), 1);
    assert_eq!(contradictions[0].edge_type, EdgeType::Contradicts);
}
