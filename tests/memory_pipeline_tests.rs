//! Use-case-level integration tests for memory ingestion and recall.
//!
//! Same shape as the item pipeline tests: a scripted chat client and a
//! deterministic mock embedder stand in for the network, while the DuckDB store
//! underneath is real. What these cover that the unit tests cannot is the part
//! where arbitration meets storage — that a *rejected* supersession really does
//! leave the prior memory answering queries, that a parked memory really does
//! vanish from recall, and that a duplicate really does not become a second row.

use std::sync::Arc;

use memory_rs::application::interfaces::Embedder;
use memory_rs::application::{MemoryIngestionUseCase, MemoryRecallUseCase, MemoryRepository};
use memory_rs::Predicate;
use memory_rs::{
    DuckdbStore, EdgeType, Entity, EntityRef, IngestionOutcome, Memory, MemoryKind, MemoryStatus,
    SessionMessage, SessionTranscript, SourceKind,
};
use openai_rs::ChatClient;

mod common;
use common::{
    ambiguous_seed_vector, embed_text, AmbiguousEmbeddingClient, ConstantEmbeddingClient,
    MockEmbeddingClient, ScriptedChatClient, DIMS,
};

const PROJECT: &str = "owner/repo";

/// The one subject every seeded memory is about.
const SUBJECT: &str = "the user";

struct Harness {
    repo: Arc<DuckdbStore>,
    chat: Arc<ScriptedChatClient>,
    /// When set, every pair embeds identically so the attribution path runs
    /// deterministically. The default hashing embedder cannot be aimed at a
    /// cosine threshold.
    constant_embeddings: bool,
    /// When set, every pair lands in the ambiguous band, where tier 3 runs.
    ambiguous_embeddings: bool,
}

impl Harness {
    fn new(responses: Vec<&str>) -> Self {
        let repo = Arc::new(DuckdbStore::in_memory(DIMS, "mock-embedding").unwrap());
        Self {
            repo,
            chat: Arc::new(ScriptedChatClient::new(responses)),
            constant_embeddings: false,
            ambiguous_embeddings: false,
        }
    }

    fn memories(&self) -> Arc<dyn MemoryRepository> {
        self.repo.clone()
    }

    fn embedder(&self) -> Embedder {
        if self.ambiguous_embeddings {
            Embedder::new(Arc::new(AmbiguousEmbeddingClient))
        } else if self.constant_embeddings {
            Embedder::new(Arc::new(ConstantEmbeddingClient))
        } else {
            Embedder::new(Arc::new(MockEmbeddingClient))
        }
    }

    fn ingestion(&self) -> MemoryIngestionUseCase {
        MemoryIngestionUseCase::new(
            self.chat.clone() as Arc<dyn ChatClient>,
            self.memories(),
            self.embedder(),
        )
    }

    fn recall(&self) -> MemoryRecallUseCase {
        MemoryRecallUseCase::new(self.memories(), self.embedder())
    }

    /// Seed an entity plus a prior memory about it, embedded so prefetch finds
    /// it. Every test here is about one subject, so the entity is fixed —
    /// varying it would add a parameter no test reads.
    async fn seed_prior(
        &self,
        memory_id: &str,
        predicate: &str,
        object: &str,
        statement: &str,
        source_kind: SourceKind,
        confidence: f32,
    ) -> String {
        let entity_name = SUBJECT;
        let entity_id = format!("entity-{entity_name}");
        self.memories()
            .upsert_entity(
                &Entity {
                    id: entity_id.clone(),
                    entity_type: "person".to_string(),
                    canonical_name: entity_name.to_string(),
                    names: Vec::new(),
                    created_at: 100,
                    updated_at: 100,
                },
                Some(&embed_text(entity_name)),
            )
            .await
            .unwrap();
        self.memories()
            .append_memory(
                &Memory {
                    id: memory_id.to_string(),
                    kind: MemoryKind::Preference,
                    subject: EntityRef::Entity(entity_id.clone()),
                    predicate: Predicate::parse(predicate).unwrap_or(Predicate::RelatesTo),
                    object: EntityRef::Literal(object.to_string()),
                    statement: statement.to_string(),
                    project: Some(PROJECT.to_string()),
                    recorded_at: 100,
                    valid_from: 100,
                    valid_to: None,
                    source_session_id: Some("session-old".to_string()),
                    source_message_index: None,
                    source_kind,
                    confidence,
                    status: MemoryStatus::Active,
                    derived: false,
                    derived_from: Vec::new(),
                },
                Some(&embed_text(statement)),
            )
            .await
            .unwrap();
        entity_id
    }
}

fn transcript(id: &str, text: &str) -> SessionTranscript {
    SessionTranscript {
        id: id.to_string(),
        source: "test".to_string(),
        project: Some(PROJECT.to_string()),
        messages: vec![SessionMessage {
            role: "user".to_string(),
            content: text.to_string(),
            timestamp: None,
        }],
    }
}

/// One extracted memory, optionally relating to `target`.
fn response(
    subject: &str,
    predicate: &str,
    object: &str,
    statement: &str,
    source_kind: &str,
    confidence: f32,
    relation: Option<(&str, &str)>,
) -> String {
    let relation = match relation {
        Some((kind, target)) => {
            format!(r#", "relation": {{"type": "{kind}", "target": "{target}"}}"#)
        }
        None => String::new(),
    };
    format!(
        r#"{{"memories": [{{"subject": "{subject}", "subject_is_entity": true,
            "subject_type": "person", "predicate": "{predicate}", "object": "{object}",
            "object_is_entity": false, "object_type": "unknown", "statement": "{statement}",
            "kind": "preference", "source_kind": "{source_kind}",
            "confidence": {confidence}{relation}}}]}}"#
    )
}

// ── Arbitration meeting storage ──────────────────────────────────────────

/// The ordinary case, and the one an earlier arbitrating version got wrong: a
/// user restates a preference, the model says it supersedes the old one, and
/// the new statement becomes the answer. When ingestion arbitrated, two
/// user-stated memories with similar confidence tied — and the tie hid *both*,
/// so the user was recalled neither time.
#[tokio::test]
async fn a_restated_preference_supersedes_the_old_one_and_stays_recallable() {
    let h = Harness::new(vec![&response(
        "the user",
        "prefers",
        "spaces",
        "the user prefers spaces",
        "user_stated",
        0.9,
        Some(("supersedes", "prior-1")),
    )]);
    h.seed_prior(
        "prior-1",
        "prefers",
        "tabs",
        "the user prefers tabs",
        SourceKind::UserStated,
        0.9, // identical confidence: the old code's tie case
    )
    .await;

    let IngestionOutcome::Ingested(report) = h
        .ingestion()
        .execute(&transcript("session-1", "indentation talk"), false)
        .await
        .unwrap()
    else {
        panic!("expected an ingest");
    };
    assert_eq!(report.memories_superseded, 1);
    assert_eq!(report.conflicts_recorded, 0);

    let hits = h
        .recall()
        .execute("indentation preference", None, None, 10)
        .await
        .unwrap();
    let statements: Vec<&str> = hits.iter().map(|h| h.memory.statement.as_str()).collect();
    assert!(
        statements.contains(&"the user prefers spaces"),
        "the current preference must be recallable, got {statements:?}",
    );
    assert!(
        !statements.contains(&"the user prefers tabs"),
        "the superseded preference must not answer queries",
    );
}

/// Supersession is honoured even when the newcomer is less trusted than what it
/// replaces. That is a deliberate trade, not an oversight: the superseded
/// memory stays linked and comes back in provenance, and consolidation can
/// supersede the supersession in turn — so a wrong call here is visible and
/// correctable rather than silent and permanent.
#[tokio::test]
async fn an_inference_may_supersede_a_user_statement() {
    let h = Harness::new(vec![&response(
        "the user",
        "prefers",
        "spaces",
        "the user prefers spaces",
        "assistant_inferred",
        0.3,
        Some(("supersedes", "prior-1")),
    )]);
    h.seed_prior(
        "prior-1",
        "prefers",
        "tabs",
        "the user prefers tabs",
        SourceKind::UserStated,
        0.95,
    )
    .await;

    let IngestionOutcome::Ingested(report) = h
        .ingestion()
        .execute(&transcript("session-1", "indentation talk"), false)
        .await
        .unwrap()
    else {
        panic!("expected an ingest");
    };
    assert_eq!(report.memories_superseded, 1);

    let prior = h.memories().find_memory("prior-1").await.unwrap().unwrap();
    assert_eq!(prior.status, MemoryStatus::Superseded);
    // Still in the graph, still linked — this is what makes the trade safe.
    let edges = h.memories().edges_to("prior-1").await.unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].edge_type, EdgeType::Supersedes);
}

/// An unambiguous supersession is the one path on which ingestion retires
/// anything — and it must close the validity window when it does.
#[tokio::test]
async fn an_accepted_supersession_retires_the_prior_and_closes_its_window() {
    let h = Harness::new(vec![&response(
        "the user",
        "prefers",
        "spaces",
        "the user prefers spaces",
        "user_stated",
        0.9,
        Some(("supersedes", "prior-1")),
    )]);
    h.seed_prior(
        "prior-1",
        "prefers",
        "tabs",
        "the user prefers tabs",
        SourceKind::AssistantInferred,
        0.5,
    )
    .await;

    let IngestionOutcome::Ingested(report) = h
        .ingestion()
        .execute(&transcript("session-1", "indentation talk"), false)
        .await
        .unwrap()
    else {
        panic!("expected an ingest");
    };
    assert_eq!(report.memories_superseded, 1);
    assert_eq!(report.conflicts_recorded, 0);

    let prior = h.memories().find_memory("prior-1").await.unwrap().unwrap();
    assert_eq!(prior.status, MemoryStatus::Superseded);
    assert!(
        prior.valid_to.is_some(),
        "a superseded memory records when it stopped being true",
    );

    let edges = h.memories().edges_to("prior-1").await.unwrap();
    assert_eq!(edges[0].edge_type, EdgeType::Supersedes);
}

/// A contradiction records the disagreement and hides nothing. Both memories
/// stay active and both keep answering, because the honest response to a
/// contested question is "these two things are on record and they disagree",
/// not silence. Consolidation reconciles them later.
#[tokio::test]
async fn a_contradiction_leaves_both_memories_recallable() {
    let h = Harness::new(vec![&response(
        "the user",
        "prefers",
        "spaces",
        "the user prefers spaces",
        "user_stated",
        0.55,
        Some(("contradicts", "prior-1")),
    )]);
    h.seed_prior(
        "prior-1",
        "prefers",
        "tabs",
        "the user prefers tabs",
        SourceKind::UserStated,
        0.5,
    )
    .await;

    let IngestionOutcome::Ingested(report) = h
        .ingestion()
        .execute(&transcript("session-1", "indentation talk"), false)
        .await
        .unwrap()
    else {
        panic!("expected an ingest");
    };
    assert_eq!(report.memories_superseded, 0);
    assert_eq!(report.conflicts_recorded, 1);

    let prior = h.memories().find_memory("prior-1").await.unwrap().unwrap();
    assert_eq!(prior.status, MemoryStatus::Active);
    assert_eq!(
        prior.valid_to, None,
        "a contradiction fixes no moment at which the memory stopped being true",
    );

    // Both sides answer. Neither is hidden pending resolution.
    let hits = h
        .recall()
        .execute("indentation preference", None, None, 10)
        .await
        .unwrap();
    let statements: Vec<&str> = hits.iter().map(|h| h.memory.statement.as_str()).collect();
    assert!(
        statements.contains(&"the user prefers tabs"),
        "{statements:?}"
    );
    assert!(
        statements.contains(&"the user prefers spaces"),
        "{statements:?}"
    );

    // And every memory in the store is active — there is no parked state to
    // get stuck in.
    let all = h.memories().list_memories(None, None, None).await.unwrap();
    assert_eq!(all.len(), 2);
    assert!(all.iter().all(|m| m.status == MemoryStatus::Active));
}

// ── Duplicate collapse (C5) ──────────────────────────────────────────────

/// Ingestion appends up to 32 memories a session and only links when the model
/// volunteers a relation. Without this collapse, the same topic across fifty
/// sessions becomes fifty near-identical active memories. It is also what makes
/// `corroborates` a live edge rather than one the model happens to emit.
#[tokio::test]
async fn a_duplicate_triple_corroborates_the_prior_instead_of_appending() {
    let h = Harness::new(vec![&response(
        "the user",
        "prefers",
        "tabs",
        "the user prefers tabs, restated",
        "user_stated",
        0.8,
        None, // no relation offered — the collapse must be automatic
    )]);
    h.seed_prior(
        "prior-1",
        "prefers",
        "tabs",
        "the user prefers tabs",
        SourceKind::UserStated,
        0.5,
    )
    .await;

    let IngestionOutcome::Ingested(report) = h
        .ingestion()
        .execute(&transcript("session-1", "indentation talk"), false)
        .await
        .unwrap()
    else {
        panic!("expected an ingest");
    };
    assert_eq!(
        report.memories_written, 0,
        "the duplicate must not become a second memory",
    );
    assert_eq!(report.memories_corroborated, 1);

    let all = h.memories().list_memories(None, None, None).await.unwrap();
    assert_eq!(
        all.len(),
        1,
        "store grew despite the memory being a duplicate"
    );

    let prior = h.memories().find_memory("prior-1").await.unwrap().unwrap();
    assert!(
        (prior.confidence - 0.55).abs() < 1e-5,
        "corroboration should nudge confidence 0.50 -> 0.55, got {}",
        prior.confidence,
    );
}

/// The collapse keys on the resolved triple, not the statement text, so a
/// genuinely different fact about the same subject still lands as its own memory.
#[tokio::test]
async fn a_different_predicate_about_the_same_subject_is_not_collapsed() {
    let h = Harness::new(vec![&response(
        "the user",
        "avoids",
        "tabs",
        "the user avoids tabs",
        "user_stated",
        0.8,
        None,
    )]);
    h.seed_prior(
        "prior-1",
        "prefers",
        "tabs",
        "the user prefers tabs",
        SourceKind::UserStated,
        0.5,
    )
    .await;

    let IngestionOutcome::Ingested(report) = h
        .ingestion()
        .execute(&transcript("session-1", "indentation talk"), false)
        .await
        .unwrap()
    else {
        panic!("expected an ingest");
    };
    assert_eq!(report.memories_written, 1);
    assert_eq!(report.memories_corroborated, 0);
    assert_eq!(
        h.memories()
            .list_memories(None, None, None)
            .await
            .unwrap()
            .len(),
        2
    );
}

// ── Anti-hallucination guard ─────────────────────────────────────────────

/// A relation naming an id the model was never shown is discarded outright.
/// Honouring it would let a hallucinated id retire an arbitrary memory.
#[tokio::test]
async fn a_relation_targeting_an_unseen_id_is_ignored() {
    let h = Harness::new(vec![&response(
        "the user",
        "prefers",
        "spaces",
        "the user prefers spaces",
        "user_stated",
        0.9,
        Some(("supersedes", "an-id-the-model-invented")),
    )]);
    h.seed_prior(
        "prior-1",
        "prefers",
        "tabs",
        "the user prefers tabs",
        SourceKind::AssistantInferred,
        0.1,
    )
    .await;

    let IngestionOutcome::Ingested(report) = h
        .ingestion()
        .execute(&transcript("session-1", "indentation talk"), false)
        .await
        .unwrap()
    else {
        panic!("expected an ingest");
    };
    assert_eq!(report.memories_written, 1);
    assert_eq!(report.edges_added, 0, "an invented target must add no edge");
    assert_eq!(report.memories_superseded, 0);
    assert_eq!(
        h.memories()
            .find_memory("prior-1")
            .await
            .unwrap()
            .unwrap()
            .status,
        MemoryStatus::Active,
    );
}

// ── Idempotence and re-ingest ────────────────────────────────────────────

#[tokio::test]
async fn re_ingesting_without_force_is_skipped_and_costs_no_model_call() {
    let h = Harness::new(vec![&response(
        "the user",
        "prefers",
        "tabs",
        "the user prefers tabs",
        "user_stated",
        0.9,
        None,
    )]);
    let t = transcript("session-1", "indentation talk");

    h.ingestion().execute(&t, false).await.unwrap();
    let calls_after_first = h.chat.recorded_calls().await.len();

    let second = h.ingestion().execute(&t, false).await.unwrap();
    assert!(matches!(second, IngestionOutcome::AlreadyIngested));
    assert_eq!(
        h.chat.recorded_calls().await.len(),
        calls_after_first,
        "a skipped re-ingest must not burn a model call",
    );
}

/// A forced re-ingest wipes the session's prior memories first, so the store
/// reflects the new run rather than accumulating both.
#[tokio::test]
async fn a_forced_re_ingest_replaces_the_sessions_memories() {
    let first = response(
        "the user",
        "prefers",
        "tabs",
        "the user prefers tabs",
        "user_stated",
        0.9,
        None,
    );
    let second = response(
        "the user",
        "prefers",
        "spaces",
        "the user prefers spaces",
        "user_stated",
        0.9,
        None,
    );
    let h = Harness::new(vec![&first, &second]);
    let t = transcript("session-1", "indentation talk");

    h.ingestion().execute(&t, false).await.unwrap();
    h.ingestion().execute(&t, true).await.unwrap();

    let all = h.memories().list_memories(None, None, None).await.unwrap();
    assert_eq!(
        all.len(),
        1,
        "the forced re-ingest should have replaced, not added"
    );
    assert_eq!(all[0].statement, "the user prefers spaces");
}

// ── Entity resolution ────────────────────────────────────────────────────

/// Two sessions naming the same subject must land on ONE entity, otherwise the
/// graph fragments and the duplicate collapse above never fires.
#[tokio::test]
async fn the_same_subject_across_sessions_resolves_to_one_entity() {
    let first = response(
        "the user",
        "prefers",
        "tabs",
        "the user prefers tabs",
        "user_stated",
        0.9,
        None,
    );
    let second = response(
        "The User",
        "uses",
        "rust",
        "the user uses rust",
        "user_stated",
        0.9,
        None,
    );
    let h = Harness::new(vec![&first, &second]);

    h.ingestion()
        .execute(&transcript("session-1", "first talk"), false)
        .await
        .unwrap();
    h.ingestion()
        .execute(&transcript("session-2", "second talk"), false)
        .await
        .unwrap();

    let entities = h.memories().list_entities().await.unwrap();
    assert_eq!(
        entities.len(),
        1,
        "name resolution is case-insensitive, so 'The User' must reuse 'the user'",
    );

    let memories = h.memories().list_memories(None, None, None).await.unwrap();
    assert_eq!(memories.len(), 2);
    let subject_ids: std::collections::HashSet<&str> = memories
        .iter()
        .filter_map(|c| c.subject.entity_id())
        .collect();
    assert_eq!(
        subject_ids.len(),
        1,
        "both memories must share one subject entity"
    );
}

/// The model supplies `subject_type`; the entity must record it, because
/// consolidation refuses to merge entities whose non-`unknown` types differ.
#[tokio::test]
async fn the_models_entity_type_is_recorded_rather_than_hardcoded() {
    let h = Harness::new(vec![&response(
        "the user",
        "prefers",
        "tabs",
        "the user prefers tabs",
        "user_stated",
        0.9,
        None,
    )]);
    h.ingestion()
        .execute(&transcript("session-1", "talk"), false)
        .await
        .unwrap();

    let entities = h.memories().list_entities().await.unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(
        entities[0].entity_type, "person",
        "a hardcoded 'unknown' would block every future entity merge",
    );
}

// ── Recall ───────────────────────────────────────────────────────────────

/// Recall walks enrichment edges one hop, and the neighbour it pulls in ranks
/// below the anchor that found it.
#[tokio::test]
async fn recall_expands_over_enrichment_edges_below_the_anchor() {
    let h = Harness::new(vec![]);
    h.seed_prior(
        "anchor-1",
        "prefers",
        "tabs",
        "the user prefers tabs for indentation",
        SourceKind::UserStated,
        0.9,
    )
    .await;
    h.seed_prior(
        "neighbour-1",
        "prefers",
        "width-4",
        "a completely unrelated topic about widths",
        SourceKind::UserStated,
        0.9,
    )
    .await;
    h.memories()
        .add_edge(&memory_rs::MemoryEdge {
            from_memory: "neighbour-1".to_string(),
            to_memory: "anchor-1".to_string(),
            edge_type: EdgeType::Refines,
            created_at: 100,
            created_by: memory_rs::EdgeOrigin::Ingestion,
            confidence: 0.9,
        })
        .await
        .unwrap();

    let hits = h
        .recall()
        .execute("tabs for indentation", None, None, 10)
        .await
        .unwrap();
    let ids: Vec<&str> = hits.iter().map(|h| h.memory.id.as_str()).collect();
    assert!(ids.contains(&"anchor-1"));
    assert!(
        ids.contains(&"neighbour-1"),
        "the refines neighbour should have been pulled in by expansion",
    );

    let anchor_score = hits
        .iter()
        .find(|h| h.memory.id == "anchor-1")
        .unwrap()
        .score;
    let neighbour_score = hits
        .iter()
        .find(|h| h.memory.id == "neighbour-1")
        .unwrap()
        .score;
    assert!(
        neighbour_score < anchor_score,
        "an expanded neighbour must rank below its anchor ({neighbour_score} vs {anchor_score})",
    );
}

/// An enrichment edge may legitimately point at a memory that has since been
/// superseded — the edge is history. Expansion must not follow it back into the
/// results, or the graph walk quietly resurrects exactly what supersession
/// retired. The two search legs filter on `status` in SQL, so expansion is the
/// one path that could reintroduce a stale memory.
#[tokio::test]
async fn expansion_does_not_resurrect_a_superseded_neighbour() {
    let h = Harness::new(vec![]);
    h.seed_prior(
        "anchor-1",
        "prefers",
        "tabs",
        "the user prefers tabs for indentation",
        SourceKind::UserStated,
        0.9,
    )
    .await;
    h.seed_prior(
        "retired-1",
        "prefers",
        "width-4",
        "a completely unrelated topic about widths",
        SourceKind::UserStated,
        0.9,
    )
    .await;
    h.memories()
        .add_edge(&memory_rs::MemoryEdge {
            from_memory: "retired-1".to_string(),
            to_memory: "anchor-1".to_string(),
            edge_type: EdgeType::Refines,
            created_at: 100,
            created_by: memory_rs::EdgeOrigin::Ingestion,
            confidence: 0.9,
        })
        .await
        .unwrap();

    // Sanity: while active, the neighbour IS pulled in — so the assertion
    // below is about the status filter, not about expansion being broken.
    let before = h
        .recall()
        .execute("tabs for indentation", None, None, 10)
        .await
        .unwrap();
    assert!(before.iter().any(|h| h.memory.id == "retired-1"));

    h.memories()
        .set_memory_status("retired-1", MemoryStatus::Superseded, Some(200))
        .await
        .unwrap();

    let after = h
        .recall()
        .execute("tabs for indentation", None, None, 10)
        .await
        .unwrap();
    let ids: Vec<&str> = after.iter().map(|h| h.memory.id.as_str()).collect();
    assert!(ids.contains(&"anchor-1"));
    assert!(
        !ids.contains(&"retired-1"),
        "expansion followed an edge back to a superseded memory",
    );
}

/// The same guard for a retracted neighbour — a memory marked as never having
/// been true must not come back through an edge either.
#[tokio::test]
async fn expansion_does_not_surface_a_retracted_neighbour() {
    let h = Harness::new(vec![]);
    h.seed_prior(
        "anchor-1",
        "prefers",
        "tabs",
        "the user prefers tabs for indentation",
        SourceKind::UserStated,
        0.9,
    )
    .await;
    h.seed_prior(
        "retracted-1",
        "prefers",
        "width-4",
        "a completely unrelated topic about widths",
        SourceKind::UserStated,
        0.9,
    )
    .await;
    h.memories()
        .add_edge(&memory_rs::MemoryEdge {
            from_memory: "retracted-1".to_string(),
            to_memory: "anchor-1".to_string(),
            edge_type: EdgeType::Corroborates,
            created_at: 100,
            created_by: memory_rs::EdgeOrigin::Ingestion,
            confidence: 0.9,
        })
        .await
        .unwrap();
    h.memories()
        .set_memory_status("retracted-1", MemoryStatus::Retracted, None)
        .await
        .unwrap();

    let hits = h
        .recall()
        .execute("tabs for indentation", None, None, 10)
        .await
        .unwrap();
    let ids: Vec<&str> = hits.iter().map(|h| h.memory.id.as_str()).collect();
    assert!(ids.contains(&"anchor-1"));
    assert!(!ids.contains(&"retracted-1"));
}

#[tokio::test]
async fn recall_filters_by_kind_and_rejects_an_empty_query() {
    let h = Harness::new(vec![]);
    h.seed_prior(
        "memory-1",
        "prefers",
        "tabs",
        "the user prefers tabs",
        SourceKind::UserStated,
        0.9,
    )
    .await;

    // Seeded memories are `preference`.
    let matching = h
        .recall()
        .execute("tabs", Some(MemoryKind::Preference), None, 10)
        .await
        .unwrap();
    assert_eq!(matching.len(), 1);

    let other_kind = h
        .recall()
        .execute("tabs", Some(MemoryKind::Skill), None, 10)
        .await
        .unwrap();
    assert!(other_kind.is_empty());

    assert!(h.recall().execute("   ", None, None, 10).await.is_err());
}

/// A namespace resolves to several projects, so scope is a slice. Globals are
/// always included; an unrelated project is not.
#[tokio::test]
async fn recall_scopes_across_several_projects_plus_globals() {
    let h = Harness::new(vec![]);
    for (id, project, statement) in [
        ("c-a", Some("owner/a"), "project a uses tabs"),
        ("c-b", Some("owner/b"), "project b uses tabs"),
        ("c-c", Some("owner/c"), "project c uses tabs"),
        ("c-global", None::<&str>, "everything everywhere uses tabs"),
    ] {
        h.memories()
            .append_memory(
                &Memory {
                    id: id.to_string(),
                    kind: MemoryKind::Fact,
                    subject: EntityRef::Literal("project".to_string()),
                    predicate: Predicate::Uses,
                    object: EntityRef::Literal("tabs".to_string()),
                    statement: statement.to_string(),
                    project: project.map(str::to_string),
                    recorded_at: 100,
                    valid_from: 100,
                    valid_to: None,
                    source_session_id: None,
                    source_message_index: None,
                    source_kind: SourceKind::UserStated,
                    confidence: 0.9,
                    status: MemoryStatus::Active,
                    derived: false,
                    derived_from: Vec::new(),
                },
                Some(&embed_text(statement)),
            )
            .await
            .unwrap();
    }

    let scope = vec!["owner/a".to_string(), "owner/b".to_string()];
    let hits = h
        .recall()
        .execute("uses tabs", None, Some(&scope), 10)
        .await
        .unwrap();
    let ids: std::collections::HashSet<&str> = hits.iter().map(|h| h.memory.id.as_str()).collect();
    assert!(ids.contains("c-a"));
    assert!(ids.contains("c-b"));
    assert!(ids.contains("c-global"), "globals are always in scope");
    assert!(
        !ids.contains("c-c"),
        "an unlisted project must stay out of scope"
    );

    // An empty slice is globals only — distinct from `None`, which is all.
    let globals_only = h
        .recall()
        .execute("uses tabs", None, Some(&[]), 10)
        .await
        .unwrap();
    let ids: Vec<&str> = globals_only.iter().map(|h| h.memory.id.as_str()).collect();
    assert_eq!(ids, ["c-global"]);
}

// ── Prompt wiring ────────────────────────────────────────────────────────

/// The prefetched prior must actually reach the model with its id, or the
/// relation machinery has nothing to point at.
#[tokio::test]
async fn prior_memories_reach_the_model_with_their_ids() {
    let h = Harness::new(vec![r#"{"memories": []}"#]);
    h.seed_prior(
        "prior-1",
        "prefers",
        "tabs",
        "the user prefers tabs",
        SourceKind::UserStated,
        0.9,
    )
    .await;

    h.ingestion()
        .execute(&transcript("session-1", "the user prefers tabs"), false)
        .await
        .unwrap();

    let calls = h.chat.recorded_calls().await;
    assert_eq!(calls.len(), 1);
    let (_, user) = &calls[0];
    assert!(
        user.contains("prior-1"),
        "the prefetched memory's id must be in the prompt:\n{user}",
    );
    assert!(user.contains("the user prefers tabs"));
}

/// An unparseable response is retried exactly once, then fails loudly rather
/// than silently storing nothing.
#[tokio::test]
async fn an_unparseable_response_is_retried_once_then_reported() {
    let h = Harness::new(vec!["not json at all", "still not json"]);
    let err = h
        .ingestion()
        .execute(&transcript("session-1", "talk"), false)
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("twice"),
        "expected a two-attempt parse failure, got: {err}",
    );
    assert_eq!(h.chat.recorded_calls().await.len(), 2);
}

#[tokio::test]
async fn a_recovered_response_on_the_retry_is_applied() {
    let good = response(
        "the user",
        "prefers",
        "tabs",
        "the user prefers tabs",
        "user_stated",
        0.9,
        None,
    );
    let h = Harness::new(vec!["not json at all", &good]);
    let IngestionOutcome::Ingested(report) = h
        .ingestion()
        .execute(&transcript("session-1", "talk"), false)
        .await
        .unwrap()
    else {
        panic!("expected an ingest");
    };
    assert_eq!(report.memories_written, 1);
    assert_eq!(h.chat.recorded_calls().await.len(), 2);
}

// ── Provenance ───────────────────────────────────────────────────────────

/// Build a supersession chain `head -> ... -> tail` by seeding each memory and
/// linking it to its predecessor. Returns the head's id.
async fn seed_chain(h: &Harness, statements: &[&str]) -> String {
    let mut previous: Option<String> = None;
    for (i, statement) in statements.iter().enumerate() {
        let id = format!("chain-{i}");
        h.seed_prior(
            &id,
            "prefers",
            statement,
            statement,
            SourceKind::UserStated,
            0.9,
        )
        .await;
        if let Some(prior) = previous {
            h.memories()
                .add_edge(&memory_rs::MemoryEdge {
                    from_memory: id.clone(),
                    to_memory: prior.clone(),
                    edge_type: EdgeType::Supersedes,
                    created_at: 100 + i as i64,
                    created_by: memory_rs::EdgeOrigin::Ingestion,
                    confidence: 0.9,
                })
                .await
                .unwrap();
            h.memories()
                .set_memory_status(&prior, MemoryStatus::Superseded, Some(100 + i as i64))
                .await
                .unwrap();
        }
        previous = Some(id);
    }
    previous.unwrap()
}

/// The whole point of the graph: a recalled memory comes back with the history
/// it replaced, so an answer can be argued with rather than just asserted.
#[tokio::test]
async fn recall_returns_the_supersession_chain_behind_a_memory() {
    let h = Harness::new(vec![]);
    seed_chain(
        &h,
        &[
            "the user prefers tabs for indentation",
            "the user prefers two spaces for indentation",
            "the user prefers four spaces for indentation",
        ],
    )
    .await;

    let hits = h
        .recall()
        .execute("indentation", None, None, 10)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1, "only the head of the chain is active");
    let hit = &hits[0];
    assert_eq!(hit.memory.id, "chain-2");

    let replaced: Vec<&str> = hit
        .provenance
        .supersedes
        .iter()
        .map(|r| r.statement.as_str())
        .collect();
    assert_eq!(
        replaced.len(),
        2,
        "expected both older links, got {replaced:?}"
    );
    assert!(replaced.iter().any(|s| s.contains("two spaces")));
    assert!(replaced.iter().any(|s| s.contains("tabs")));
    assert!(!hit.provenance.chain_truncated);
}

/// A chain deeper than the walk says so rather than presenting a partial
/// history as if it were complete.
#[tokio::test]
async fn a_chain_deeper_than_the_walk_is_reported_as_truncated() {
    let h = Harness::new(vec![]);
    let statements: Vec<String> = (0..9)
        .map(|i| format!("the user prefers indentation style {i}"))
        .collect();
    let refs: Vec<&str> = statements.iter().map(String::as_str).collect();
    seed_chain(&h, &refs).await;

    let hits = h
        .recall()
        .execute("indentation style", None, None, 10)
        .await
        .unwrap();
    let hit = &hits[0];
    assert!(
        hit.provenance.chain_truncated,
        "a 9-link chain must report truncation",
    );
    assert!(
        hit.provenance.supersedes.len() <= 5,
        "the walk must stay bounded, got {}",
        hit.provenance.supersedes.len(),
    );
}

/// A cycle in the supersession edges must terminate the walk rather than spin.
/// Nothing stops a model emitting A supersedes B and later B supersedes A.
#[tokio::test]
async fn a_supersession_cycle_does_not_hang_the_walk() {
    let h = Harness::new(vec![]);
    h.seed_prior(
        "cycle-a",
        "prefers",
        "tabs",
        "the user prefers tabs for indentation",
        SourceKind::UserStated,
        0.9,
    )
    .await;
    h.seed_prior(
        "cycle-b",
        "prefers",
        "spaces",
        "the user prefers spaces for indentation",
        SourceKind::UserStated,
        0.9,
    )
    .await;
    for (from, to) in [("cycle-a", "cycle-b"), ("cycle-b", "cycle-a")] {
        h.memories()
            .add_edge(&memory_rs::MemoryEdge {
                from_memory: from.to_string(),
                to_memory: to.to_string(),
                edge_type: EdgeType::Supersedes,
                created_at: 100,
                created_by: memory_rs::EdgeOrigin::Ingestion,
                confidence: 0.9,
            })
            .await
            .unwrap();
    }

    let hits = h
        .recall()
        .execute("indentation", None, None, 10)
        .await
        .unwrap();
    assert_eq!(hits.len(), 2);
    for hit in &hits {
        assert!(
            hit.provenance.supersedes.len() <= 1,
            "a cycle must not be walked twice",
        );
    }
}

/// A live disagreement travels with the answer. Both sides are returned, and
/// each carries a pointer to the other.
#[tokio::test]
async fn recall_reports_a_live_contradiction_on_both_sides() {
    let h = Harness::new(vec![]);
    h.seed_prior(
        "left-1",
        "prefers",
        "tabs",
        "the user prefers tabs for indentation",
        SourceKind::UserStated,
        0.9,
    )
    .await;
    h.seed_prior(
        "right-1",
        "prefers",
        "spaces",
        "the user prefers spaces for indentation",
        SourceKind::UserStated,
        0.9,
    )
    .await;
    h.memories()
        .add_edge(&memory_rs::MemoryEdge {
            from_memory: "right-1".to_string(),
            to_memory: "left-1".to_string(),
            edge_type: EdgeType::Contradicts,
            created_at: 100,
            created_by: memory_rs::EdgeOrigin::Ingestion,
            confidence: 0.9,
        })
        .await
        .unwrap();

    let hits = h
        .recall()
        .execute("indentation", None, None, 10)
        .await
        .unwrap();
    assert_eq!(hits.len(), 2, "both sides of a disagreement still answer");
    for hit in &hits {
        assert_eq!(
            hit.provenance.contradicted_by.len(),
            1,
            "{} should report the other side",
            hit.memory.id,
        );
    }

    // Once one side is superseded the disagreement is settled, and reporting it
    // would be misleading.
    h.memories()
        .set_memory_status("left-1", MemoryStatus::Superseded, Some(200))
        .await
        .unwrap();
    let hits = h
        .recall()
        .execute("indentation", None, None, 10)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].provenance.contradicted_by.is_empty());
}

/// Corroboration is a count, not a list — repetition is weak evidence and
/// listing every restatement would crowd out the answer.
#[tokio::test]
async fn recall_counts_corroborations() {
    let h = Harness::new(vec![]);
    h.seed_prior(
        "anchor-1",
        "prefers",
        "tabs",
        "the user prefers tabs for indentation",
        SourceKind::UserStated,
        0.9,
    )
    .await;
    for i in 0..3 {
        let id = format!("echo-{i}");
        h.seed_prior(
            &id,
            "prefers",
            "tabs",
            &format!("restatement number {i} about something else entirely"),
            SourceKind::UserStated,
            0.5,
        )
        .await;
        h.memories()
            .add_edge(&memory_rs::MemoryEdge {
                from_memory: id,
                to_memory: "anchor-1".to_string(),
                edge_type: EdgeType::Corroborates,
                created_at: 100 + i,
                created_by: memory_rs::EdgeOrigin::Ingestion,
                confidence: 0.9,
            })
            .await
            .unwrap();
    }

    let hits = h
        .recall()
        .execute("tabs for indentation", None, None, 10)
        .await
        .unwrap();
    let anchor = hits
        .iter()
        .find(|h| h.memory.id == "anchor-1")
        .expect("anchor should be recalled");
    assert_eq!(anchor.provenance.corroborations, 3);
}

/// A memory with no history carries an empty provenance, so surfaces can skip
/// rendering it entirely rather than showing a row of zeroes.
#[tokio::test]
async fn a_memory_with_no_history_has_empty_provenance() {
    let h = Harness::new(vec![]);
    h.seed_prior(
        "lonely-1",
        "prefers",
        "tabs",
        "the user prefers tabs for indentation",
        SourceKind::UserStated,
        0.9,
    )
    .await;

    let hits = h
        .recall()
        .execute("indentation", None, None, 10)
        .await
        .unwrap();
    assert!(hits[0].provenance.is_empty());
}

// ── Entity attribution ───────────────────────────────────────────────────

/// The vector `ConstantEmbeddingClient` produces, for seeding entities that
/// must match whatever ingestion embeds.
fn constant_vector() -> Vec<f32> {
    let mut v = vec![0.0f32; DIMS];
    v[0] = 1.0;
    v
}

/// A harness whose embedder scores every pair at 1.0, so the attribution path
/// runs deterministically. The thresholds themselves are unit-tested; this
/// covers the wiring — that a fuzzy hit reuses the entity and teaches it the
/// surface form.
fn attributing_harness(responses: Vec<&str>) -> Harness {
    let mut h = Harness::new(responses);
    h.constant_embeddings = true;
    h
}

/// The fragmentation this prevents: without a fuzzy tier, every surface variant
/// of one thing becomes a permanent separate anchor, and memories about one are
/// invisible from the other.
///
/// The variant here is an abbreviation on purpose. Role-word variants ("the
/// orders-events service") no longer reach this tier at all — the normalized
/// name key settles them for free — so testing one here would prove nothing
/// about the fuzzy path.
#[tokio::test]
async fn a_variant_surface_form_is_attributed_to_the_existing_entity() {
    let h = attributing_harness(vec![&response(
        "ord-events",
        "uses",
        "terraform",
        "ord-events uses terraform",
        "user_stated",
        0.9,
        None,
    )]);
    h.memories()
        .upsert_entity(
            &Entity {
                id: "entity-ge".into(),
                entity_type: "person".into(),
                canonical_name: "orders-events".into(),
                names: Vec::new(),
                created_at: 1,
                updated_at: 1,
            },
            Some(&constant_vector()),
        )
        .await
        .unwrap();

    let IngestionOutcome::Ingested(report) = h
        .ingestion()
        .execute(&transcript("session-1", "talk"), false)
        .await
        .unwrap()
    else {
        panic!("expected an ingest");
    };

    assert_eq!(
        report.entities_attributed, 1,
        "the variant should have resolved"
    );
    let entities = h.memories().list_entities().await.unwrap();
    assert_eq!(
        entities.len(),
        1,
        "one thing should mean one anchor, got {:?}",
        entities
            .iter()
            .map(|e| &e.canonical_name)
            .collect::<Vec<_>>()
    );

    // The half that compounds: the variant is written back, so the next
    // sighting takes the free exact path instead of re-paying for a search.
    let learned = h
        .memories()
        .find_entity("entity-ge")
        .await
        .unwrap()
        .unwrap();
    assert!(
        learned.names.iter().any(|a| a == "ord-events"),
        "attribution must teach the entity its new surface form, got {:?}",
        learned.names,
    );
}

/// The duplicate this prevents, taken from a real store: one session recorded
/// the same service as "orders-events package" and "orders-events service"
/// and produced two anchors. Neither of the other tiers could catch it — the
/// names are not equal, and at 0.907 cosine the pair sits below the attribute
/// threshold, so the decision fell to a small local model that said "different".
///
/// It is settled here for free, before any embedding or model call: both names
/// normalize to the same key.
#[tokio::test]
async fn a_role_word_variant_resolves_on_the_name_tier_without_a_model_call() {
    // No fuzzy help: this must pass on the name key alone.
    let h = Harness::new(vec![&response(
        "orders-events package",
        "uses",
        "terraform",
        "the orders-events package uses terraform",
        "user_stated",
        0.9,
        None,
    )]);
    h.memories()
        .upsert_entity(
            &Entity {
                id: "entity-ge".into(),
                entity_type: "person".into(),
                canonical_name: "the orders-events service".into(),
                names: Vec::new(),
                created_at: 1,
                updated_at: 1,
            },
            None,
        )
        .await
        .unwrap();

    let IngestionOutcome::Ingested(report) = h
        .ingestion()
        .execute(&transcript("session-1", "talk"), false)
        .await
        .unwrap()
    else {
        panic!("expected an ingest");
    };

    let entities = h.memories().list_entities().await.unwrap();
    assert_eq!(
        entities.len(),
        1,
        "one service, one anchor — got {:?}",
        entities
            .iter()
            .map(|e| &e.canonical_name)
            .collect::<Vec<_>>()
    );
    assert_eq!(report.entities_created, 0, "no second anchor was minted");
    assert_eq!(
        report.entity_adjudications, 0,
        "the name tier must settle this, not the model",
    );
}

/// A type conflict must survive even a perfect embedding match, because an
/// entity has no supersession chain to undo a bad merge with.
#[tokio::test]
async fn attribution_refuses_to_cross_entity_types() {
    let h = attributing_harness(vec![&response(
        "orders-events",
        "uses",
        "terraform",
        "orders-events uses terraform",
        "user_stated",
        0.9,
        None,
    )]);
    // The extraction fixture types its subject `person`; seed a `tool` of the
    // same name so the only thing keeping them apart is the type guard.
    h.memories()
        .upsert_entity(
            &Entity {
                id: "entity-tool".into(),
                entity_type: "tool".into(),
                canonical_name: "a completely different name".into(),
                names: Vec::new(),
                created_at: 1,
                updated_at: 1,
            },
            Some(&constant_vector()),
        )
        .await
        .unwrap();

    let IngestionOutcome::Ingested(report) = h
        .ingestion()
        .execute(&transcript("session-1", "talk"), false)
        .await
        .unwrap()
    else {
        panic!("expected an ingest");
    };
    assert_eq!(
        report.entities_attributed, 0,
        "a type conflict must not attribute"
    );
    assert_eq!(h.memories().list_entities().await.unwrap().len(), 2);
}

/// The reverse direction: memories point at entities, so "what do we know about
/// this thing" is a lookup back through both FK columns — subject *and* object.
#[tokio::test]
async fn memories_for_entity_finds_it_as_subject_and_as_object() {
    let h = Harness::new(vec![]);
    h.memories()
        .upsert_entity(
            &Entity {
                id: "entity-x".into(),
                entity_type: "tool".into(),
                canonical_name: "Terraform".into(),
                names: Vec::new(),
                created_at: 1,
                updated_at: 1,
            },
            None,
        )
        .await
        .unwrap();

    let mut as_subject = seeded_memory("m-subject", "Terraform provides state locking");
    as_subject.subject = EntityRef::Entity("entity-x".into());
    let mut as_object = seeded_memory("m-object", "the deployment uses Terraform");
    as_object.object = EntityRef::Entity("entity-x".into());
    for m in [&as_subject, &as_object] {
        h.memories().append_memory(m, None).await.unwrap();
    }

    let found = h.memories().memories_for_entity("entity-x").await.unwrap();
    let ids: std::collections::HashSet<&str> = found.iter().map(|m| m.id.as_str()).collect();
    assert!(ids.contains("m-subject"), "missed the subject reference");
    assert!(ids.contains("m-object"), "missed the object reference");
}

/// A bare memory for the reverse-lookup test, with no entity refs of its own.
fn seeded_memory(id: &str, statement: &str) -> Memory {
    Memory {
        id: id.to_string(),
        kind: MemoryKind::Fact,
        subject: EntityRef::Literal("something".into()),
        predicate: Predicate::Uses,
        object: EntityRef::Literal("something else".into()),
        statement: statement.to_string(),
        project: None,
        recorded_at: 100,
        valid_from: 100,
        valid_to: None,
        source_session_id: Some("session-1".into()),
        source_message_index: None,
        source_kind: SourceKind::UserStated,
        confidence: 0.9,
        status: MemoryStatus::Active,
        derived: false,
        derived_from: Vec::new(),
    }
}

// ── Ambiguous-band adjudication (tier 3) ─────────────────────────────────

/// A name in the ambiguous band is settled by asking the model. This is the
/// case neither of the cheaper tiers can fix: an abbreviation shares no
/// normalized key with the name it abbreviates, and it lands too far below the
/// attribute threshold to merge on the score — while lowering that threshold
/// would start merging things that are genuinely distinct.
#[tokio::test]
async fn an_ambiguous_name_is_merged_when_adjudication_says_same() {
    let extraction = response(
        "orders-service",
        "uses",
        "terraform",
        "the orders-service uses terraform",
        "user_stated",
        0.9,
        None,
    );
    // Scripted responses are consumed in order: extraction, then adjudication.
    let h = ambiguous_harness(vec![&extraction, r#"{"same": true}"#]);
    seed_ambiguous_candidate(&h).await;

    let IngestionOutcome::Ingested(report) = h
        .ingestion()
        .execute(&transcript("session-1", "talk"), false)
        .await
        .unwrap()
    else {
        panic!("expected an ingest");
    };

    assert_eq!(
        report.entities_ambiguous, 1,
        "the band should have been hit"
    );
    assert_eq!(report.entity_adjudications, 1, "and adjudicated");
    assert_eq!(report.entities_adjudicated_same, 1);
    assert_eq!(
        h.memories().list_entities().await.unwrap().len(),
        1,
        "a `same` verdict must collapse the duplicate anchor"
    );
}

/// A `false` verdict keeps them apart — related-but-distinct is the common case
/// and the one a bare threshold gets wrong in the other direction.
#[tokio::test]
async fn an_ambiguous_name_stays_separate_when_adjudication_says_different() {
    let extraction = response(
        "orders-service",
        "uses",
        "terraform",
        "the orders-service uses terraform",
        "user_stated",
        0.9,
        None,
    );
    let h = ambiguous_harness(vec![&extraction, r#"{"same": false}"#]);
    seed_ambiguous_candidate(&h).await;

    let IngestionOutcome::Ingested(report) = h
        .ingestion()
        .execute(&transcript("session-1", "talk"), false)
        .await
        .unwrap()
    else {
        panic!("expected an ingest");
    };
    assert_eq!(report.entity_adjudications, 1);
    assert_eq!(report.entities_adjudicated_same, 0);
    assert_eq!(h.memories().list_entities().await.unwrap().len(), 2);
}

/// The failure that must never happen: an unreachable or broken model merging
/// two entities by default. Merging cannot be undone — every memory anchored to
/// either side silently becomes a memory about a thing that does not exist —
/// so an absent answer has to mean "keep them apart".
#[tokio::test]
async fn a_failed_adjudication_never_merges() {
    let extraction = response(
        "orders-service",
        "uses",
        "terraform",
        "the orders-service uses terraform",
        "user_stated",
        0.9,
        None,
    );
    // No second scripted response: the adjudication call errors.
    let h = ambiguous_harness(vec![&extraction]);
    seed_ambiguous_candidate(&h).await;

    let IngestionOutcome::Ingested(report) = h
        .ingestion()
        .execute(&transcript("session-1", "talk"), false)
        .await
        .unwrap()
    else {
        panic!("expected an ingest");
    };
    assert_eq!(
        report.entity_adjudications, 1,
        "the attempt is still counted"
    );
    assert_eq!(report.entities_adjudicated_same, 0);
    assert_eq!(
        h.memories().list_entities().await.unwrap().len(),
        2,
        "a failed adjudication must leave a recoverable duplicate, not a merge"
    );
}

/// An embedder that places every pair inside the ambiguous band (0.85–0.95),
/// which no hashing embedder can be aimed at.
fn ambiguous_harness(responses: Vec<&str>) -> Harness {
    let mut h = Harness::new(responses);
    h.ambiguous_embeddings = true;
    h
}

async fn seed_ambiguous_candidate(h: &Harness) {
    h.memories()
        .upsert_entity(
            &Entity {
                id: "entity-ge".into(),
                entity_type: "person".into(),
                canonical_name: "orders-svc".into(),
                names: Vec::new(),
                created_at: 1,
                updated_at: 1,
            },
            Some(&ambiguous_seed_vector()),
        )
        .await
        .unwrap();
}
