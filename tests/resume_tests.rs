//! Tests for the resume briefing — "what was I working on".
//!
//! The store is real (in-memory DuckDB) and no model is involved: a briefing is
//! assembled from what import already wrote, which is the property that makes it
//! cheap enough to run at the start of every session. If any of these ever need
//! a chat client, the use case has grown a dependency it should not have.

use std::sync::Arc;

use memory_rs::application::MemoryRepository;
use memory_rs::domain::{EntityRef, Memory, MemoryKind, MemoryNode, MemoryStatus, SourceKind};
use memory_rs::Predicate;
use memory_rs::{
    DuckdbStore, ImportedSession, MemoryResumeUseCase, NodeKind, NodeRepository, SessionStatus,
};

const DIMS: usize = 4;

fn store() -> Arc<DuckdbStore> {
    Arc::new(DuckdbStore::in_memory(DIMS, "mock-embedding").unwrap())
}

fn use_case(store: &Arc<DuckdbStore>) -> MemoryResumeUseCase {
    MemoryResumeUseCase::new(store.clone(), store.clone())
}

fn session(id: &str, project: Option<&str>, imported_at: i64) -> ImportedSession {
    ImportedSession {
        id: id.to_string(),
        source: "zed".to_string(),
        imported_at,
        message_count: 40,
        project: project.map(str::to_string),
        items_written: 2,
        status: SessionStatus::Imported,
        last_error: None,
    }
}

fn session_node(id: &str, summary: &str, overview: &str) -> MemoryNode {
    MemoryNode::new(
        format!("memory://sessions/{id}"),
        NodeKind::Session,
        Some("memory://sessions".to_string()),
        summary.to_string(),
        overview.to_string(),
        "the full transcript".to_string(),
        1,
        1,
    )
}

fn memory(id: &str, session_id: &str, project: Option<&str>, statement: &str) -> Memory {
    Memory {
        id: id.to_string(),
        kind: MemoryKind::Fact,
        subject: EntityRef::Literal("orders-events".to_string()),
        predicate: Predicate::Uses,
        object: EntityRef::Literal("kafka".to_string()),
        statement: statement.to_string(),
        project: project.map(str::to_string),
        recorded_at: 100,
        valid_from: 100,
        valid_to: None,
        source_session_id: Some(session_id.to_string()),
        source_message_index: None,
        source_kind: SourceKind::UserStated,
        confidence: 0.9,
        status: MemoryStatus::Active,
        derived: false,
        derived_from: Vec::new(),
    }
}

/// The whole point: one call returns the recent sessions, newest first, each
/// carrying the summary written at import *and* the memories it left behind.
#[tokio::test]
async fn a_briefing_pairs_each_session_with_its_summary_and_its_memories() {
    let store = store();
    store
        .record_session(&session("older", Some("owner/repo"), 100))
        .await
        .unwrap();
    store
        .record_session(&session("newer", Some("owner/repo"), 200))
        .await
        .unwrap();
    store
        .upsert_node(
            &session_node(
                "newer",
                "Wired the queue consumer to the database.",
                "- goal\n- result",
            ),
            None,
        )
        .await
        .unwrap();
    store
        .append_memory(
            &memory(
                "m-1",
                "newer",
                Some("owner/repo"),
                "orders-events uses Kafka",
            ),
            None,
        )
        .await
        .unwrap();
    store
        .append_memory(
            &memory("m-2", "older", Some("owner/repo"), "an older fact"),
            None,
        )
        .await
        .unwrap();

    let briefing = use_case(&store).execute(None, 10).await.unwrap();

    let ids: Vec<&str> = briefing
        .sessions
        .iter()
        .map(|r| r.session.id.as_str())
        .collect();
    assert_eq!(ids, ["newer", "older"], "newest work must come first");

    let newest = &briefing.sessions[0];
    assert_eq!(newest.summary, "Wired the queue consumer to the database.");
    assert!(newest.overview.contains("- goal"));
    assert_eq!(
        newest
            .memories
            .iter()
            .map(|m| m.id.as_str())
            .collect::<Vec<_>>(),
        ["m-1"],
        "a session's memories must not leak into another's recap",
    );

    // A session with no node still appears — it was still work. It just has no
    // summary, which is honest rather than absent from the briefing.
    let oldest = &briefing.sessions[1];
    assert!(oldest.summary.is_empty());
    assert_eq!(oldest.memories.len(), 1);
    assert_eq!(briefing.more, 0);
}

/// Scoping is the feature: opening a project must not brief you on every other
/// project you touched this week.
#[tokio::test]
async fn scoping_to_a_project_excludes_other_projects_and_unattributed_sessions() {
    let store = store();
    store
        .record_session(&session("mine", Some("owner/repo"), 300))
        .await
        .unwrap();
    store
        .record_session(&session("theirs", Some("other/repo"), 200))
        .await
        .unwrap();
    // Imported before the project column existed: it cannot be attributed, so it
    // must not be guessed into someone's briefing.
    store
        .record_session(&session("legacy", None, 100))
        .await
        .unwrap();

    let scoped = use_case(&store)
        .execute(Some(&["owner/repo".to_string()]), 10)
        .await
        .unwrap();
    assert_eq!(
        scoped
            .sessions
            .iter()
            .map(|r| r.session.id.as_str())
            .collect::<Vec<_>>(),
        ["mine"],
    );
    assert_eq!(scoped.projects, vec!["owner/repo".to_string()]);

    // Unscoped, everything shows — including the one that cannot be attributed.
    let all = use_case(&store).execute(None, 10).await.unwrap();
    assert_eq!(all.sessions.len(), 3);
    assert!(all.projects.is_empty());
}

/// A failed harvest is a "do not retry" marker with no transcript, no summary
/// and no memories. Briefing on it would push real work out of the window.
#[tokio::test]
async fn failed_harvest_markers_are_not_work() {
    let store = store();
    let mut failed = session("broken", Some("owner/repo"), 300);
    failed.status = SessionStatus::Failed;
    failed.last_error = Some("could not parse transcript".to_string());
    store.record_session(&failed).await.unwrap();
    store
        .record_session(&session("real", Some("owner/repo"), 100))
        .await
        .unwrap();

    let briefing = use_case(&store).execute(None, 10).await.unwrap();
    assert_eq!(
        briefing
            .sessions
            .iter()
            .map(|r| r.session.id.as_str())
            .collect::<Vec<_>>(),
        ["real"],
    );
}

/// The limit bounds the payload, and `more` says how much was left out — a
/// silently truncated briefing reads as "that's all there was".
#[tokio::test]
async fn the_limit_is_honoured_and_the_remainder_is_reported() {
    let store = store();
    for i in 0..5 {
        store
            .record_session(&session(&format!("s-{i}"), Some("owner/repo"), i))
            .await
            .unwrap();
    }

    let briefing = use_case(&store).execute(None, 2).await.unwrap();
    assert_eq!(briefing.sessions.len(), 2);
    assert_eq!(briefing.more, 3);

    // A zero or absurd limit is clamped rather than returning nothing or
    // everything.
    assert_eq!(
        use_case(&store)
            .execute(None, 0)
            .await
            .unwrap()
            .sessions
            .len(),
        1
    );
    assert_eq!(
        use_case(&store)
            .execute(None, 10_000)
            .await
            .unwrap()
            .sessions
            .len(),
        5
    );
}

#[tokio::test]
async fn an_empty_store_briefs_empty_rather_than_erroring() {
    let store = store();
    let briefing = use_case(&store).execute(None, 5).await.unwrap();
    assert!(briefing.is_empty());
    assert_eq!(briefing.more, 0);
}
