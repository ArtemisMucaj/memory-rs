//! Use-case-level integration tests for the import → ingest → store → recall
//! pipeline, driven by test doubles for the LLM and embedding backends.
//!
//! These exercise the same use case the CLI router and the HTTP API call, but
//! wire it up with a scripted chat client and a deterministic mock embedder
//! instead of a live model — so the pipeline runs offline and reproducibly. The
//! DuckDB store is real (in-memory).
//!
//! Scope note: `memory_pipeline_tests.rs` covers ingestion *itself* (arbitration,
//! duplicate collapse, entity resolution). What this file adds is the part
//! `ImportSessionUseCase` owns on top of ingestion — the minimum-length gate,
//! the session marker, and that the memories a session writes are afterwards
//! recallable.

use std::sync::Arc;

use memory_rs::application::interfaces::Embedder;
use memory_rs::application::{MemoryIngestionUseCase, MemoryRecallUseCase, MemoryRepository};
use memory_rs::{
    DuckdbStore, ImportOutcome, ImportSessionUseCase, NodeRepository, SessionMessage,
    SessionTranscript, SummarizeMemoryUseCase,
};

use openai_rs::ChatClient;

mod common;
use common::{MockEmbeddingClient, ScriptedChatClient, DIMS};

struct Harness {
    repo: Arc<DuckdbStore>,
    embedder: Embedder,
}

impl Harness {
    fn new() -> Self {
        Self {
            repo: Arc::new(DuckdbStore::in_memory(DIMS, "mock-embedding").unwrap()),
            embedder: Embedder::new(Arc::new(MockEmbeddingClient)),
        }
    }

    fn node_repo(&self) -> Arc<dyn NodeRepository> {
        self.repo.clone()
    }

    fn memories(&self) -> Arc<dyn MemoryRepository> {
        self.repo.clone()
    }

    fn import_use_case(&self, chat: Arc<ScriptedChatClient>) -> ImportSessionUseCase {
        let ingestion = MemoryIngestionUseCase::new(
            Arc::clone(&chat) as Arc<dyn ChatClient>,
            self.memories(),
            self.embedder.clone(),
        );
        let summary = SummarizeMemoryUseCase::new(
            chat as Arc<dyn ChatClient>,
            self.node_repo(),
            self.memories(),
            Arc::new(self.embedder.clone()),
        );
        ImportSessionUseCase::new(self.node_repo(), ingestion, summary)
    }

    fn recall_use_case(&self) -> MemoryRecallUseCase {
        MemoryRecallUseCase::new(self.memories(), self.embedder.clone())
    }
}

fn transcript(id: &str, messages: &[(&str, &str)]) -> SessionTranscript {
    SessionTranscript {
        id: id.to_string(),
        source: format!("{id}.jsonl"),
        project: None,
        messages: messages
            .iter()
            .map(|(role, content)| SessionMessage {
                role: role.to_string(),
                content: content.to_string(),
                timestamp: Some("2026-07-01T10:00:00Z".to_string()),
            })
            .collect(),
    }
}

/// One extracted memory, as the ingestion model would return it.
fn memory_json(predicate: &str, object: &str, statement: &str) -> String {
    format!(
        r#"{{"memories": [{{"subject": "the user", "subject_is_entity": true,
            "subject_type": "person", "predicate": "{predicate}", "object": "{object}",
            "object_is_entity": false, "object_type": "unknown", "statement": "{statement}",
            "kind": "preference", "source_kind": "user_stated", "confidence": 0.9}}]}}"#
    )
}

#[tokio::test]
async fn import_ingests_memories_stores_and_records_session() {
    let harness = Harness::new();
    let chat = Arc::new(ScriptedChatClient::new(vec![
        r###"{"memories": [
            {"subject": "the user", "subject_is_entity": true, "subject_type": "person",
             "predicate": "prefers", "object": "the ? operator", "object_is_entity": false,
             "object_type": "unknown",
             "statement": "the user prefers ? over unwrap in library code",
             "kind": "preference", "source_kind": "user_stated", "confidence": 0.95},
            {"subject": "the project", "subject_is_entity": true, "subject_type": "project",
             "predicate": "stores_data_in", "object": "an embedded column store",
             "object_is_entity": false, "object_type": "unknown",
             "statement": "the project stores indexed data in an embedded column store",
             "kind": "fact", "source_kind": "assistant_inferred", "confidence": 0.8}
        ]}"###,
    ]));
    let use_case = harness.import_use_case(Arc::clone(&chat));

    let transcript = transcript(
        "session-1",
        &[
            (
                "user",
                "Please never use unwrap in library code, use ? instead",
            ),
            ("assistant", "Understood, refactored to use ? everywhere."),
        ],
    );
    let outcome = use_case.execute(&transcript, false).await.unwrap();

    let ImportOutcome::Imported { session, report } = outcome else {
        panic!("expected Imported outcome");
    };
    assert_eq!(session.id, "session-1");
    assert_eq!(report.memories_written, 2);
    assert_eq!(
        session.items_written, 2,
        "the session marker records memories"
    );

    let memories = harness
        .memories()
        .list_memories(None, None, None)
        .await
        .unwrap();
    assert_eq!(memories.len(), 2);
    assert!(memories
        .iter()
        .any(|c| c.statement.contains("? over unwrap in library code")));

    // Both subjects were resolved into entities rather than left as literals.
    assert_eq!(harness.memories().list_entities().await.unwrap().len(), 2);

    // The session marker is recorded and the ingestion prompt carried the chat.
    let recorded = harness
        .node_repo()
        .find_session("session-1")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(recorded.message_count, 2);
    let calls = chat.recorded_calls().await;
    assert_eq!(calls.len(), 1);
    assert!(calls[0].0.contains("atomic MEMORIES"));
    assert!(calls[0].1.contains("never use unwrap"));
}

#[tokio::test]
async fn import_is_idempotent_unless_forced() {
    let harness = Harness::new();
    let chat = Arc::new(ScriptedChatClient::new(vec![
        &memory_json("prefers", "tabs", "the user prefers tabs"),
        &memory_json(
            "prefers",
            "tabs",
            "the user prefers tabs, and feels strongly about it",
        ),
    ]));
    let use_case = harness.import_use_case(chat);
    let transcript = transcript(
        "session-2",
        &[("user", "I prefer tabs"), ("assistant", "Noted.")],
    );

    assert!(matches!(
        use_case.execute(&transcript, false).await.unwrap(),
        ImportOutcome::Imported { .. }
    ));
    // Second import without force is skipped (no scripted response consumed).
    assert!(matches!(
        use_case.execute(&transcript, false).await.unwrap(),
        ImportOutcome::AlreadyImported { .. }
    ));
    // A forced re-import replaces the session's memories rather than appending a
    // second near-identical set beside them.
    assert!(matches!(
        use_case.execute(&transcript, true).await.unwrap(),
        ImportOutcome::Imported { .. }
    ));

    let memories = harness
        .memories()
        .list_memories(None, None, None)
        .await
        .unwrap();
    assert_eq!(
        memories.len(),
        1,
        "forced re-import should replace, not add"
    );
    assert!(memories[0].statement.contains("feels strongly"));
}

#[tokio::test]
async fn imported_memories_are_recallable() {
    let harness = Harness::new();
    let chat = Arc::new(ScriptedChatClient::new(vec![
        r###"{"memories": [
        {"subject": "the project", "subject_is_entity": true, "subject_type": "project",
         "predicate": "retries", "object": "network timeouts", "object_is_entity": false,
         "object_type": "unknown",
         "statement": "retry network timeouts with exponential backoff",
         "kind": "fact", "source_kind": "user_stated", "confidence": 0.9}]}"###,
    ]));
    let use_case = harness.import_use_case(chat);
    let transcript = transcript(
        "session-3",
        &[
            ("user", "how should we handle network timeouts?"),
            ("assistant", "retry with backoff"),
        ],
    );
    use_case.execute(&transcript, false).await.unwrap();

    // Hybrid recall (semantic via the mock embedder + keyword) finds it.
    let results = harness
        .recall_use_case()
        .execute("network timeout", None, None, 10)
        .await
        .unwrap();
    assert!(!results.is_empty(), "expected a hit for the stored memory");
    assert!(results[0].memory.statement.contains("network timeouts"));
}

/// A transcript too short to be worth a model call is rejected before one is
/// made — the gate `ImportSessionUseCase` owns on top of ingestion.
#[tokio::test]
async fn a_too_short_transcript_is_rejected_without_a_model_call() {
    let harness = Harness::new();
    let chat = Arc::new(ScriptedChatClient::new(vec![]));
    let use_case = harness.import_use_case(Arc::clone(&chat));

    // `ImportOutcome` is not `Debug`, so match rather than `unwrap_err`.
    match use_case
        .execute(&transcript("session-4", &[("user", "hi")]), false)
        .await
    {
        Err(e) => assert!(e.to_string().contains("minimum"), "unexpected error: {e}"),
        Ok(_) => panic!("a one-message transcript should have been rejected"),
    }
    assert!(chat.recorded_calls().await.is_empty());
}
