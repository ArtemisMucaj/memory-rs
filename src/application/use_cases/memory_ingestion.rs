//! Memory ingestion — the online write path for the memory graph.
//!
//! Flow (one bounded LLM call with a format-recovery retry):
//!
//! 1. **Idempotence / re-import** — a non-forced re-ingest of a session that
//!    already produced memories is skipped; a forced one hard-deletes the
//!    session's prior memories first. That delete is the single sanctioned
//!    destructive operation in the system: re-running extraction over an
//!    unchanged transcript is a do-over, not a new observation.
//! 2. **Prefetch** — semantic-search existing active memories for context, so the
//!    model can *relate* a new memory to a prior one instead of duplicating it.
//! 3. **Extract** — one call returns atomic subject–predicate–object memories,
//!    each with an optional typed relation to a prefetched memory.
//! 4. **Apply** — resolve entities, collapse duplicates, record relations,
//!    and append.
//!
//! # What step 4 does not do
//!
//! *Ingestion never weighs two memories against each other.*
//!
//! It has the least context of any pass — one session, a handful of prefetched
//! neighbours, a small local model — and it is the only pass that runs
//! unsupervised on every import. So it records what the model asserts and
//! defers every judgement:
//!
//! - **`supersedes` is honoured unconditionally.** Supersession is a temporal
//!   chain: the prior memory *was* true, the new one is true now, and the newest
//!   link is the current answer. There is nothing to arbitrate. An earlier
//!   version did arbitrate here, comparing source kinds and confidences, and
//!   that was the bug — an ordinary preference update ("tabs", later "spaces",
//!   both user-stated with similar confidence) tied, and a tie hid *both*
//!   memories, so a user who had said what they wanted twice was recalled
//!   neither time.
//! - **`contradicts` is recorded and nothing else happens.** Both memories stay
//!   active and both keep answering. Hiding them would look cautious and be
//!   dishonest: the true answer to a contested question is that two things are
//!   on record and they disagree. Consolidation, which sees the whole
//!   neighbourhood, reconciles the pair into a new memory that supersedes both.
//!
//! What makes deferring safe is that nothing here is silent or final. A
//! superseded memory stays in the graph, linked, and comes back in a result's
//! provenance; consolidation can supersede a supersession in turn. The cost of
//! a wrong call is a visible, correctable edge rather than a memory that
//! vanished.
//!
//! This is the append-only counterpart to
//! [`MemoryExtractionUseCase`](super::MemoryExtractionUseCase): it never
//! rewrites a memory in place.

use std::collections::HashMap;
use std::sync::Arc;

use serde::Deserialize;
use tracing::{debug, warn};
use uuid::Uuid;

use openai_rs::ChatClient;

use crate::application::interfaces::{Embedder, MemoryRepository};
use crate::application::use_cases::llm_json::{
    extract_json_object, repair_json_string_escapes, unix_now,
};
use crate::application::use_cases::memory_ingestion_prompt as prompt;
use crate::domain::{
    entity_name_key, DomainError, EdgeOrigin, EdgeType, Entity, EntityRef, Memory, MemoryEdge,
    MemoryKind, MemoryStatus, Predicate, SessionTranscript, SourceKind,
};

/// How many prior memories are prefetched into the extraction context.
const PREFETCH_LIMIT: usize = 8;

/// Upper bound on memories applied from a single ingestion, guarding against a
/// runaway model flooding the store.
const MAX_MEMORIES_PER_RUN: usize = 32;

/// Confidence added to a memory each time an independent session corroborates
/// it. Small on purpose: corroboration is weak evidence (the same user saying
/// the same thing twice is not two witnesses), and a larger step would let
/// repetition alone push a memory past [`CONFIDENCE_MARGIN`] and win future
/// arbitrations outright.
///
/// [`CONFIDENCE_MARGIN`]: crate::domain::CONFIDENCE_MARGIN
const CORROBORATION_BUMP: f32 = 0.05;

/// How many entity candidates the similarity search considers. Only the best
/// is ever used; the rest exist so a type conflict on the top hit does not hide
/// a correct second one from the logs.
const ENTITY_CANDIDATES: usize = 5;

/// Cosine similarity at or above which a surface form is attributed to an
/// existing entity outright.
///
/// Deliberately high. A wrong attribution silently merges two distinct things
/// and there is no supersession chain to undo it with — unlike a memory, an
/// entity has no history. Missing a match merely leaves a duplicate for the
/// consolidation pass to merge later, which is the cheaper mistake.
const ATTRIBUTE_THRESHOLD: f32 = 0.95;

/// Below [`ATTRIBUTE_THRESHOLD`] but at or above this, a match is close enough
/// that a human might call it the same entity. Nothing is done — the case is
/// only counted, as evidence for whether an adjudication step is worth adding.
const AMBIGUOUS_THRESHOLD: f32 = 0.85;

/// What the similarity search's best candidate warrants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Attribution {
    /// Confident enough: reuse the entity and teach it this surface form.
    Attribute,
    /// Close enough to be arguable, not close enough to act on. Counted, so the
    /// size of this band is the evidence for whether an adjudication step would
    /// earn its cost.
    Ambiguous,
    /// Different things.
    Distinct,
}

/// Decide what a candidate match warrants. Pure, so the thresholds and the type
/// guard can be tested exhaustively without a store or an embedder — which
/// matters because a hash-based test embedder cannot produce a meaningful
/// cosine score near the boundary.
/// Whether two entity types are different enough that the entities must be
/// kept apart.
///
/// Never merge across two entities the model typed differently. A wrong merge
/// is unrecoverable — an entity carries no supersession chain the way a memory
/// does — so this errs toward keeping them apart. `unknown` matches anything,
/// because it asserts nothing.
///
/// Shared by both resolution tiers: the name tier needs it because
/// [`entity_name_key`] is broader than the name it came from, and the
/// similarity tier because a close cosine score says nothing about type.
pub(crate) fn types_conflict(existing_type: &str, new_type: &str) -> bool {
    existing_type != new_type
        && existing_type != UNKNOWN_ENTITY_TYPE
        && new_type != UNKNOWN_ENTITY_TYPE
}

pub(crate) fn attribution_for(score: f32, existing_type: &str, new_type: &str) -> Attribution {
    if types_conflict(existing_type, new_type) {
        return Attribution::Distinct;
    }
    if score >= ATTRIBUTE_THRESHOLD {
        Attribution::Attribute
    } else if score >= AMBIGUOUS_THRESHOLD {
        Attribution::Ambiguous
    } else {
        Attribution::Distinct
    }
}

/// Most adjudication calls one ingestion will make.
///
/// Each is a model round-trip on the write path, so the band that reaches this
/// tier has to stay small. Beyond the cap the run falls back to treating the
/// remaining ambiguities as distinct — a duplicate entity a later pass can
/// merge, which is the recoverable failure.
const MAX_ADJUDICATIONS_PER_RUN: usize = 8;

/// Default entity type when the model does not supply a usable one.
///
/// `unknown` is load-bearing rather than a null: consolidation refuses to merge
/// two entities whose non-`unknown` types differ, so a *wrong* guess here
/// permanently blocks a legitimate merge, while `unknown` merely declines to
/// help.
const UNKNOWN_ENTITY_TYPE: &str = "unknown";

/// What one ingestion produced.
#[derive(Debug, Default, PartialEq)]
pub struct IngestionReport {
    pub memories_written: usize,
    pub entities_created: usize,
    pub edges_added: usize,
    /// Surface forms attributed to an existing entity by similarity rather than
    /// by an exact name match — the fragmentation this prevented.
    pub entities_attributed: usize,
    /// Near-misses: close enough to be arguable, not close enough to attribute
    /// on the score alone. Each one reaching the cap becomes an adjudication.
    pub entities_ambiguous: usize,
    /// Adjudication calls made — the cost this tier actually incurred.
    pub entity_adjudications: usize,
    /// Adjudications that came back "same thing", so a duplicate anchor was
    /// avoided. The ratio against `entity_adjudications` says whether the
    /// ambiguous band is worth calling a model over at all.
    pub entities_adjudicated_same: usize,
    /// Memories whose predicate fell outside the closed vocabulary and landed
    /// on `relates_to`. The metric for whether [`Predicate`] needs extending.
    pub predicates_out_of_vocabulary: usize,
    /// Contradictions recorded. Both sides stay active and keep answering; the
    /// disagreement is consolidation's to reconcile.
    pub conflicts_recorded: usize,
    /// Prior memories corroborated instead of duplicated (both the model's own
    /// `corroborates` relations and duplicates collapsed by
    /// [`MemoryIngestionUseCase`]'s normalization).
    pub memories_corroborated: usize,
    /// Prior memories retired by an unambiguous supersession.
    pub memories_superseded: usize,
}

/// Outcome of an ingestion request.
#[derive(Debug)]
pub enum IngestionOutcome {
    /// Extraction ran; the report describes what was written.
    Ingested(IngestionReport),
    /// The session already had memories and `force` was not set.
    AlreadyIngested,
}

/// JSON shape the extraction model must return (mirrors [`prompt::schema`]).
#[derive(Debug, Deserialize)]
struct RawIngestion {
    #[serde(default)]
    memories: Vec<RawMemory>,
}

#[derive(Debug, Deserialize)]
struct RawMemory {
    #[serde(default)]
    subject: String,
    #[serde(default)]
    subject_is_entity: bool,
    #[serde(default)]
    subject_type: String,
    #[serde(default)]
    predicate: String,
    #[serde(default)]
    object: String,
    #[serde(default)]
    object_is_entity: bool,
    #[serde(default)]
    object_type: String,
    #[serde(default)]
    statement: String,
    #[serde(default)]
    kind: String,
    #[serde(default)]
    source_kind: String,
    #[serde(default)]
    confidence: f32,
    #[serde(default)]
    relation: Option<RawRelation>,
}

#[derive(Debug, Deserialize)]
struct RawRelation {
    #[serde(default, rename = "type")]
    kind: String,
    #[serde(default)]
    target: String,
}

/// What ingestion does about one new memory and its relation target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Verdict {
    /// The edge actually written — not always the one the model asked for.
    edge_type: EdgeType,
    /// What happens to the prior memory.
    prior: PriorAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PriorAction {
    /// Left exactly as it was.
    Untouched,
    /// Retired with a closed validity window: it is no longer the head of its
    /// supersession chain.
    Supersede,
    /// Confidence nudged up by [`CORROBORATION_BUMP`].
    Corroborate,
}

/// Decide the fate of a new memory and its relation target.
///
/// A pure function of the edge type — ingestion does not weigh the two memories
/// against each other at all. That is the deliberate shape of this design, and
/// it is worth saying why, because an earlier version did arbitrate here and
/// the arbitration was the bug.
///
/// **Supersession is a temporal chain, not a judgement.** `supersedes` asserts
/// that the prior memory *was* true and the new one is true now. The newest
/// link is the current answer, so there is nothing to decide: record the edge,
/// retire the prior, and let the chain be the history. Arbitrating instead —
/// comparing source kinds and confidences — meant an ordinary preference update
/// ("tabs", then later "spaces", both stated by the user with similar
/// confidence) landed in a tie and hid *both* memories, so a user who had said
/// what they wanted twice got recalled neither time.
///
/// **Contradiction is the only real conflict, and this is the wrong place to
/// settle it.** Ingestion sees one session and a handful of prefetched
/// neighbours, running a small model, unsupervised, on every import.
/// Consolidation sees the whole neighbourhood. So a `contradicts` edge is
/// recorded and *nothing else happens*: both memories stay active, both keep
/// answering, and the disagreement travels with them in a result's provenance
/// until consolidation reconciles it into a new memory that supersedes both.
fn adjudicate(requested: EdgeType) -> Verdict {
    match requested {
        // Temporal replacement: the chain moves on.
        EdgeType::Supersedes => Verdict {
            edge_type: EdgeType::Supersedes,
            prior: PriorAction::Supersede,
        },
        // Recorded, unresolved, and visible. Consolidation's input.
        EdgeType::Contradicts => Verdict {
            edge_type: EdgeType::Contradicts,
            prior: PriorAction::Untouched,
        },
        // Enrichment: both remain true, nothing to do beyond the edge.
        EdgeType::Refines | EdgeType::RelatesTo => Verdict {
            edge_type: requested,
            prior: PriorAction::Untouched,
        },
        EdgeType::Corroborates => Verdict {
            edge_type: EdgeType::Corroborates,
            prior: PriorAction::Corroborate,
        },
        // Not offered by the ingestion schema; consolidation-only. Treated as a
        // navigational link rather than honoured, so a model that invents it
        // cannot assert that something was never true.
        EdgeType::Retracts => Verdict {
            edge_type: EdgeType::RelatesTo,
            prior: PriorAction::Untouched,
        },
    }
}

/// Identity used to collapse near-duplicate memories within and across sessions.
///
/// Entity refs contribute their resolved id, so two surface forms that resolved
/// to the same entity produce the same key; literals are normalized.
fn duplicate_key(memory: &Memory) -> String {
    fn part(r: &EntityRef) -> String {
        match r {
            EntityRef::Entity(id) => format!("e:{id}"),
            EntityRef::Literal(v) => format!("l:{}", v.trim().to_lowercase()),
        }
    }
    format!(
        "{}|{}|{}|{}",
        part(&memory.subject),
        // No normalization needed any more: the predicate is a closed enum, so
        // its stored form is already canonical. That is the whole point of
        // constraining it — surface drift can no longer split one fact in two.
        memory.predicate.as_str(),
        part(&memory.object),
        memory.project.as_deref().unwrap_or(""),
    )
}

pub struct MemoryIngestionUseCase {
    chat_client: Arc<dyn ChatClient>,
    memory_repo: Arc<dyn MemoryRepository>,
    embedder: Embedder,
}

impl MemoryIngestionUseCase {
    pub fn new(
        chat_client: Arc<dyn ChatClient>,
        memory_repo: Arc<dyn MemoryRepository>,
        embedder: Embedder,
    ) -> Self {
        Self {
            chat_client,
            memory_repo,
            embedder,
        }
    }

    /// Ingest `transcript` into the memory graph.
    #[tracing::instrument(skip_all, fields(session_id = %transcript.id))]
    pub async fn execute(
        &self,
        transcript: &SessionTranscript,
        force: bool,
    ) -> Result<IngestionOutcome, DomainError> {
        if force {
            let removed = self
                .memory_repo
                .delete_memories_for_session(&transcript.id)
                .await?;
            if removed > 0 {
                debug!(
                    "memory ingestion: forced re-import removed {removed} prior memories for '{}'",
                    transcript.id
                );
            }
        } else if self
            .memory_repo
            .count_memories_for_session(&transcript.id)
            .await?
            > 0
        {
            return Ok(IngestionOutcome::AlreadyIngested);
        }

        let prior = self.prefetch(transcript).await;
        let raw = self.extract(transcript, &prior).await?;
        let report = self.apply(transcript, raw, &prior).await?;
        Ok(IngestionOutcome::Ingested(report))
    }

    /// Semantic-search existing active memories for context. Failures degrade to
    /// "no context" rather than failing the ingest — losing prefetch costs
    /// deduplication quality, but losing the session costs the memory itself.
    async fn prefetch(&self, transcript: &SessionTranscript) -> Vec<Memory> {
        if !self.embedder.embeddings_enabled() {
            return Vec::new();
        }
        let query = prompt::prefetch_query(transcript);
        if query.trim().is_empty() {
            return Vec::new();
        }
        // Globals plus this session's project — the same scope the memory will
        // be written into.
        //
        // A session whose project could not be resolved narrows to globals
        // alone (the empty list), never to the whole store. The distinction is
        // load-bearing: an unscoped search has no `WHERE` clause at all, so it
        // would put another project's memories in front of the model *with
        // their ids*, and a `supersedes` naming one of those ids is honoured
        // unconditionally — retiring a memory belonging to a project this
        // session has nothing to do with.
        let projects: Vec<String> = transcript.project.clone().into_iter().collect();
        match self.embedder.embed_query(&query).await {
            Ok(vector) => match self
                .memory_repo
                .search_memories_semantic(&vector, None, Some(&projects), PREFETCH_LIMIT)
                .await
            {
                Ok(results) => results.into_iter().map(|(memory, _)| memory).collect(),
                Err(e) => {
                    warn!("memory prefetch search failed: {e}");
                    Vec::new()
                }
            },
            Err(e) => {
                warn!("memory prefetch embedding failed: {e}");
                Vec::new()
            }
        }
    }

    /// Call the extraction model and parse its JSON, retrying once with a
    /// format-correction message when parsing fails.
    async fn extract(
        &self,
        transcript: &SessionTranscript,
        prior: &[Memory],
    ) -> Result<RawIngestion, DomainError> {
        let system = prompt::system_prompt();
        let user = prompt::user_prompt(transcript, prior);
        let schema = prompt::schema();
        let response = self
            .chat_client
            .complete_json(&system, &user, "memory_ingestion", &schema)
            .await?;
        match parse_ingestion(&response) {
            Ok(parsed) => Ok(parsed),
            Err(first_err) => {
                debug!("memory ingestion output unparseable, retrying once: {first_err}");
                let retry_user = format!("{user}\n\n{}", prompt::format_retry_prompt());
                let response = self
                    .chat_client
                    .complete_json(&system, &retry_user, "memory_ingestion", &schema)
                    .await?;
                parse_ingestion(&response).map_err(|e| {
                    DomainError::parse(format!(
                        "memory ingestion model returned unparseable output twice: {e}"
                    ))
                })
            }
        }
    }

    /// Resolve entities, collapse duplicates, arbitrate conflicts, append.
    async fn apply(
        &self,
        transcript: &SessionTranscript,
        raw: RawIngestion,
        prior: &[Memory],
    ) -> Result<IngestionReport, DomainError> {
        let now = unix_now();
        let prior_by_id: HashMap<&str, &Memory> =
            prior.iter().map(|c| (c.id.as_str(), c)).collect();
        let mut entity_cache: HashMap<String, String> = HashMap::new();
        let mut report = IngestionReport::default();

        // Duplicate index over the prefetched priors, plus everything written
        // in this run so a model that emits the same triple twice writes once.
        let mut seen: HashMap<String, String> = prior
            .iter()
            .map(|c| (duplicate_key(c), c.id.clone()))
            .collect();

        for raw_memory in raw.memories {
            if report.memories_written >= MAX_MEMORIES_PER_RUN {
                debug!(
                    "memory ingestion: hit the {MAX_MEMORIES_PER_RUN}-memory cap, dropping the rest"
                );
                break;
            }
            let statement = raw_memory.statement.trim();
            if statement.is_empty() {
                continue;
            }
            // A predicate outside the vocabulary falls back to the escape hatch
            // rather than dropping the memory — losing a real fact over a word
            // choice is the worse failure — but it is counted, because a rising
            // share of these is the evidence that the list needs extending.
            let predicate = match Predicate::parse(&raw_memory.predicate) {
                Some(p) => p,
                None => {
                    report.predicates_out_of_vocabulary += 1;
                    debug!(
                        "predicate '{}' is outside the vocabulary; recorded as relates_to",
                        raw_memory.predicate.trim()
                    );
                    Predicate::RelatesTo
                }
            };

            let subject = self
                .resolve_ref(
                    &raw_memory.subject,
                    raw_memory.subject_is_entity,
                    &raw_memory.subject_type,
                    statement,
                    now,
                    &mut entity_cache,
                    &mut report,
                )
                .await?;
            let object = self
                .resolve_ref(
                    &raw_memory.object,
                    raw_memory.object_is_entity,
                    &raw_memory.object_type,
                    statement,
                    now,
                    &mut entity_cache,
                    &mut report,
                )
                .await?;

            let memory = Memory {
                id: Uuid::new_v4().to_string(),
                kind: MemoryKind::parse(&raw_memory.kind).unwrap_or(MemoryKind::Fact),
                subject,
                predicate,
                object,
                statement: statement.to_string(),
                project: transcript.project.clone(),
                recorded_at: now,
                valid_from: now,
                valid_to: None,
                source_session_id: Some(transcript.id.clone()),
                source_message_index: None,
                source_kind: SourceKind::parse(&raw_memory.source_kind)
                    .unwrap_or(SourceKind::AssistantInferred),
                confidence: raw_memory.confidence.clamp(0.0, 1.0),
                status: MemoryStatus::Active,
                derived: false,
                derived_from: Vec::new(),
            };

            // Resolve the relation the model asked for, if any, honouring it
            // only when it names a memory we actually put in front of the model.
            // Without this guard a hallucinated id would attach an edge to an
            // arbitrary memory — or, worse, retire one.
            let requested: Option<(EdgeType, &Memory)> =
                raw_memory.relation.as_ref().and_then(|r| {
                    let target = r.target.trim();
                    let edge_type = EdgeType::parse(&r.kind)?;
                    let prior_memory = prior_by_id.get(target)?;
                    Some((edge_type, *prior_memory))
                });

            // No relation offered, but this triple already exists: record
            // corroboration instead of appending a near-identical memory. This
            // is what stops the same topic across fifty sessions becoming fifty
            // active memories, and it is also what makes `corroborates` a live
            // edge rather than one the model happens to volunteer.
            let key = duplicate_key(&memory);
            if requested.is_none() {
                if let Some(existing_id) = seen.get(&key).cloned() {
                    self.corroborate(&existing_id, &memory, now, &mut report)
                        .await?;
                    continue;
                }
            }

            let verdict =
                requested.map(|(edge_type, prior_memory)| (adjudicate(edge_type), prior_memory));

            // Every memory is appended `Active`. Nothing ingestion learns from
            // one session justifies withholding a memory from recall.
            let vector = self.embed_opt(&memory.statement).await;
            self.memory_repo
                .append_memory(&memory, vector.as_deref())
                .await?;
            report.memories_written += 1;
            seen.insert(key, memory.id.clone());

            let Some((verdict, prior_memory)) = verdict else {
                continue;
            };
            self.memory_repo
                .add_edge(&MemoryEdge {
                    from_memory: memory.id.clone(),
                    to_memory: prior_memory.id.clone(),
                    edge_type: verdict.edge_type,
                    created_at: now,
                    created_by: EdgeOrigin::Ingestion,
                    confidence: memory.confidence,
                })
                .await?;
            report.edges_added += 1;
            if verdict.edge_type == EdgeType::Contradicts {
                report.conflicts_recorded += 1;
            }

            match verdict.prior {
                PriorAction::Untouched => {}
                PriorAction::Supersede => {
                    self.memory_repo
                        .set_memory_status(&prior_memory.id, MemoryStatus::Superseded, Some(now))
                        .await?;
                    report.memories_superseded += 1;
                }
                PriorAction::Corroborate => {
                    self.bump_confidence(prior_memory, &mut report).await?;
                }
            }
        }
        Ok(report)
    }

    /// Record that `memory` independently confirms the memory at `existing_id`,
    /// without appending a second near-identical row.
    async fn corroborate(
        &self,
        existing_id: &str,
        memory: &Memory,
        now: i64,
        report: &mut IngestionReport,
    ) -> Result<(), DomainError> {
        let Some(existing) = self.memory_repo.find_memory(existing_id).await? else {
            return Ok(());
        };
        // Self-edges are meaningless and would make the graph cyclic at depth
        // zero; a duplicate seen twice in one run just bumps confidence.
        if existing.id != memory.id {
            self.memory_repo
                .add_edge(&MemoryEdge {
                    from_memory: memory.id.clone(),
                    to_memory: existing.id.clone(),
                    edge_type: EdgeType::Corroborates,
                    created_at: now,
                    created_by: EdgeOrigin::Ingestion,
                    confidence: memory.confidence,
                })
                .await
                // The new memory was never appended, so an edge from it would
                // dangle. Record corroboration on the surviving memory only.
                .ok();
        }
        self.bump_confidence(&existing, report).await
    }

    async fn bump_confidence(
        &self,
        prior: &Memory,
        report: &mut IngestionReport,
    ) -> Result<(), DomainError> {
        let bumped = (prior.confidence + CORROBORATION_BUMP).min(1.0);
        self.memory_repo
            .set_memory_confidence(&prior.id, bumped)
            .await?;
        report.memories_corroborated += 1;
        Ok(())
    }

    /// Resolve a subject/object surface form to an [`EntityRef`].
    ///
    /// Literals pass through. Entity references climb a ladder, cheapest first:
    ///
    /// 1. the in-run cache,
    /// 2. an exact (case-insensitive) match on a name the entity goes by,
    /// 3. a cosine search over entity-name embeddings,
    /// 4. otherwise a new entity.
    ///
    /// Step 3 is what stops the graph fragmenting. Without it, "orders-events",
    /// "the orders-events service" and "orders events" are three permanent
    /// anchors that never connect, because step 2 only ever matches a string
    /// somebody already wrote down.
    ///
    /// And when step 3 hits, the surface form is **recorded as a name**. That
    /// is the half that compounds: the variant costs one vector search once,
    /// and every later sighting is a free exact match. An attribution mechanism
    /// that does not remember its own decisions re-pays for them forever.
    #[allow(clippy::too_many_arguments)]
    async fn resolve_ref(
        &self,
        surface: &str,
        is_entity: bool,
        entity_type: &str,
        // The statement this mention appears in — context for adjudication,
        // which needs to see how a name is *used*, not just how it is spelled.
        statement: &str,
        now: i64,
        cache: &mut HashMap<String, String>,
        report: &mut IngestionReport,
    ) -> Result<EntityRef, DomainError> {
        let trimmed = surface.trim();
        if !is_entity || trimmed.is_empty() {
            return Ok(EntityRef::Literal(trimmed.to_string()));
        }
        let entity_type = match entity_type.trim() {
            "" => UNKNOWN_ENTITY_TYPE.to_string(),
            t => t.to_lowercase(),
        };
        // Keyed by the same normalization the store uses, so the two surface
        // forms one session produces for one thing ("orders-events package",
        // "the orders-events service") share a cache slot too — not just a row.
        let key = format!("{}\u{0}{entity_type}", entity_name_key(trimmed));
        if let Some(id) = cache.get(&key) {
            return Ok(EntityRef::Entity(id.clone()));
        }
        // The key is broader than the name, so a hit can be a differently-typed
        // entity that merely shares it. Take the first the type guard accepts —
        // the same rule the similarity tier applies, so both tiers agree on what
        // "the same thing" means.
        let named = self.memory_repo.find_entities_by_name(trimmed).await?;
        if let Some(existing) = named
            .into_iter()
            .find(|e| !types_conflict(&e.entity_type, &entity_type))
        {
            cache.insert(key, existing.id.clone());
            return Ok(EntityRef::Entity(existing.id));
        }

        // The embedding covers name *and* type, so "orders-events (project)"
        // and "orders-events (tool)" do not collapse into each other.
        let embedding_text = format!("{trimmed} ({entity_type})");
        let vector = self.embed_opt(&embedding_text).await;

        if let Some(vector) = vector.as_deref() {
            if let Some(existing) = self
                .attribute_by_similarity(trimmed, &entity_type, statement, vector, report)
                .await?
            {
                cache.insert(key, existing.clone());
                return Ok(EntityRef::Entity(existing));
            }
        }

        let entity = Entity {
            id: Uuid::new_v4().to_string(),
            entity_type,
            canonical_name: trimmed.to_string(),
            names: Vec::new(),
            created_at: now,
            updated_at: now,
        };
        self.memory_repo
            .upsert_entity(&entity, vector.as_deref())
            .await?;
        report.entities_created += 1;
        cache.insert(key, entity.id.clone());
        Ok(EntityRef::Entity(entity.id))
    }

    /// Attribute `surface` to an existing entity by embedding similarity,
    /// returning its id and teaching it the new name.
    ///
    /// Only a confident match attributes. The band below
    /// [`ATTRIBUTE_THRESHOLD`] but above [`AMBIGUOUS_THRESHOLD`] is where a
    /// human — or a model with more context than this path has — would have to
    /// decide, so nothing is done except *count* it. That count is the evidence
    /// for whether an adjudication step is worth adding: if the band is empty
    /// in practice, an LLM tier here would be cost with no benefit.
    #[allow(clippy::too_many_arguments)]
    async fn attribute_by_similarity(
        &self,
        surface: &str,
        entity_type: &str,
        statement: &str,
        vector: &[f32],
        report: &mut IngestionReport,
    ) -> Result<Option<String>, DomainError> {
        let candidates = self
            .memory_repo
            .search_entities_semantic(vector, ENTITY_CANDIDATES)
            .await?;
        let Some((entity, score)) = candidates.into_iter().next() else {
            return Ok(None);
        };
        match attribution_for(score, &entity.entity_type, entity_type) {
            Attribution::Attribute => {}
            Attribution::Ambiguous => {
                report.entities_ambiguous += 1;
                // Too close to call from the score alone. This is the one place
                // a model call earns its cost: the alternative is a permanent
                // duplicate anchor, and the answer gets memoized as a name, so
                // a given variant is adjudicated once ever — not once a session.
                if report.entity_adjudications >= MAX_ADJUDICATIONS_PER_RUN {
                    debug!("adjudication cap reached; '{surface}' recorded separately");
                    return Ok(None);
                }
                report.entity_adjudications += 1;
                if !self
                    .adjudicate_same_entity(surface, entity_type, &entity, statement)
                    .await
                {
                    return Ok(None);
                }
                report.entities_adjudicated_same += 1;
            }
            Attribution::Distinct => return Ok(None),
        }

        // Teach the entity the surface form that just resolved to it, so the
        // next sighting takes the free exact path.
        let mut learned = entity.clone();
        if !learned
            .names
            .iter()
            .any(|a| a.eq_ignore_ascii_case(surface))
            && !learned.canonical_name.eq_ignore_ascii_case(surface)
        {
            learned.names.push(surface.to_string());
        }
        // `upsert_entity` replaces the row and its vector, so the vector is
        // re-supplied: passing `None` would silently drop this entity out of
        // the very search that just found it.
        let existing_vector = self
            .embed_opt(&format!(
                "{} ({})",
                learned.canonical_name, learned.entity_type
            ))
            .await;
        self.memory_repo
            .upsert_entity(&learned, existing_vector.as_deref())
            .await?;
        report.entities_attributed += 1;
        Ok(Some(entity.id))
    }

    /// Ask the model whether two close-but-not-identical names are the same
    /// thing. Errors and unparseable answers resolve to **no**: an unavailable
    /// model must leave a recoverable duplicate, never merge on a guess.
    async fn adjudicate_same_entity(
        &self,
        surface: &str,
        surface_type: &str,
        candidate: &Entity,
        example: &str,
    ) -> bool {
        let system = prompt::entity_adjudication_system_prompt();
        let user = prompt::entity_adjudication_user_prompt(
            surface,
            surface_type,
            example,
            &candidate.canonical_name,
            &candidate.entity_type,
            "an earlier memory",
        );
        let schema = prompt::entity_adjudication_schema();
        let Ok(response) = self
            .chat_client
            .complete_json(&system, &user, "entity_adjudication", &schema)
            .await
        else {
            warn!("entity adjudication call failed; keeping '{surface}' separate");
            return false;
        };
        let Some(json) = extract_json_object(&response) else {
            return false;
        };
        let same = serde_json::from_str::<serde_json::Value>(json)
            .ok()
            .and_then(|v| v.get("same").and_then(serde_json::Value::as_bool))
            .unwrap_or(false);
        debug!(
            "adjudicated '{surface}' vs '{}': same={same}",
            candidate.canonical_name
        );
        same
    }

    /// Embed `text`, returning `None` when embeddings are disabled or the call
    /// fails (the row stays keyword-searchable either way).
    async fn embed_opt(&self, text: &str) -> Option<Vec<f32>> {
        if !self.embedder.embeddings_enabled() {
            return None;
        }
        match self.embedder.embed_query(text).await {
            Ok(vector) => Some(vector),
            Err(e) => {
                warn!("failed to embed memory text: {e}");
                None
            }
        }
    }
}

/// Parse the model's ingestion JSON, tolerating prose/fences and the
/// invalid-escape output small local models emit.
fn parse_ingestion(response: &str) -> Result<RawIngestion, DomainError> {
    let json = extract_json_object(response)
        .ok_or_else(|| DomainError::parse("no JSON object found in ingestion output"))?;
    match serde_json::from_str::<RawIngestion>(json) {
        Ok(parsed) => Ok(parsed),
        Err(strict_err) => {
            let repaired = repair_json_string_escapes(json);
            serde_json::from_str::<RawIngestion>(&repaired)
                .map_err(|_| DomainError::parse(format!("invalid ingestion JSON: {strict_err}")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory_with(id: &str, source_kind: SourceKind, confidence: f32) -> Memory {
        Memory {
            id: id.to_string(),
            kind: MemoryKind::Fact,
            subject: EntityRef::Entity("entity-1".to_string()),
            predicate: Predicate::Uses,
            object: EntityRef::Literal("svc-a".to_string()),
            statement: "the team uses svc-a".to_string(),
            project: Some("owner/repo".to_string()),
            recorded_at: 100,
            valid_from: 100,
            valid_to: None,
            source_session_id: Some("session-1".to_string()),
            source_message_index: None,
            source_kind,
            confidence,
            status: MemoryStatus::Active,
            derived: false,
            derived_from: Vec::new(),
        }
    }

    // ── The §7 table ─────────────────────────────────────────────────────

    /// Enrichment relations leave both memories current.
    #[test]
    fn refines_and_relates_to_leave_both_active() {
        for edge in [EdgeType::Refines, EdgeType::RelatesTo] {
            let v = adjudicate(edge);
            assert_eq!(v.edge_type, edge);
            assert_eq!(v.prior, PriorAction::Untouched);
        }
    }

    #[test]
    fn corroborates_bumps_the_prior_and_retires_nothing() {
        let v = adjudicate(EdgeType::Corroborates);
        assert_eq!(v.edge_type, EdgeType::Corroborates);
        assert_eq!(v.prior, PriorAction::Corroborate);
    }

    /// Supersession is honoured unconditionally — no comparison of source kinds
    /// or confidences. The regression this guards is concrete: when ingestion
    /// *did* arbitrate, an ordinary preference update ("tabs", later "spaces",
    /// both stated by the user with similar confidence) tied, and the tie hid
    /// both memories, so a user who said what they wanted twice was recalled
    /// neither time.
    #[test]
    fn supersedes_always_retires_the_prior_whatever_the_sources() {
        let kinds = [
            SourceKind::UserStated,
            SourceKind::AssistantInferred,
            SourceKind::Derived,
        ];
        for new_kind in kinds {
            for prior_kind in kinds {
                for (new_conf, prior_conf) in [(0.9, 0.1), (0.5, 0.5), (0.1, 0.9)] {
                    let _new = memory_with("new", new_kind, new_conf);
                    let _prior = memory_with("old", prior_kind, prior_conf);
                    let v = adjudicate(EdgeType::Supersedes);
                    assert_eq!(v.edge_type, EdgeType::Supersedes);
                    assert_eq!(
                        v.prior,
                        PriorAction::Supersede,
                        "supersession must not depend on {new_kind:?}/{prior_kind:?} \
                         or on confidence {new_conf}/{prior_conf}",
                    );
                }
            }
        }
    }

    /// A contradiction is recorded and nothing else happens. Both memories stay
    /// active and keep answering; consolidation reconciles them later into a new
    /// memory that supersedes both. Ingestion sees one session and a handful of
    /// neighbours — it is the wrong place to pick a winner.
    #[test]
    fn contradicts_records_the_disagreement_and_touches_neither_memory() {
        let v = adjudicate(EdgeType::Contradicts);
        assert_eq!(v.edge_type, EdgeType::Contradicts);
        assert_eq!(
            v.prior,
            PriorAction::Untouched,
            "a contradiction must not retire or hide either side",
        );
    }

    /// `retracts` is consolidation-only. If a model emits it anyway it must not
    /// become a retraction — ingestion has no authority to say a memory was
    /// never true.
    #[test]
    fn retracts_from_ingestion_is_defanged_to_a_navigational_link() {
        let v = adjudicate(EdgeType::Retracts);
        assert_eq!(v.edge_type, EdgeType::RelatesTo);
        assert_eq!(v.prior, PriorAction::Untouched);
    }

    /// Across the whole table, `supersedes` is the only relation that retires
    /// anything. This is the rule the module doc states; if a future edit breaks
    /// it, this is the test that should fail.
    #[test]
    fn only_supersession_ever_retires_a_memory() {
        for edge in [
            EdgeType::Supersedes,
            EdgeType::Contradicts,
            EdgeType::Refines,
            EdgeType::Corroborates,
            EdgeType::Retracts,
            EdgeType::RelatesTo,
        ] {
            if adjudicate(edge).prior == PriorAction::Supersede {
                assert_eq!(edge, EdgeType::Supersedes);
            }
        }
    }

    // ── Entity attribution ───────────────────────────────────────────────

    /// The thresholds, exhaustively. A hash-based test embedder cannot produce
    /// a controlled cosine score, so the boundary is pinned here rather than
    /// through the store.
    #[test]
    fn attribution_thresholds() {
        // Confident: reuse.
        assert_eq!(
            attribution_for(1.0, "project", "project"),
            Attribution::Attribute
        );
        assert_eq!(
            attribution_for(ATTRIBUTE_THRESHOLD, "project", "project"),
            Attribution::Attribute,
            "the threshold is inclusive",
        );
        // Arguable: counted, not acted on.
        assert_eq!(
            attribution_for(ATTRIBUTE_THRESHOLD - 0.01, "project", "project"),
            Attribution::Ambiguous,
        );
        assert_eq!(
            attribution_for(AMBIGUOUS_THRESHOLD, "project", "project"),
            Attribution::Ambiguous,
        );
        // Unrelated.
        assert_eq!(
            attribution_for(AMBIGUOUS_THRESHOLD - 0.01, "project", "project"),
            Attribution::Distinct,
        );
        assert_eq!(
            attribution_for(0.0, "project", "project"),
            Attribution::Distinct
        );
    }

    /// A type conflict outranks any score. Merging two differently-typed
    /// entities cannot be undone — there is no supersession chain for an
    /// entity — so even a perfect name match must not do it.
    #[test]
    fn a_type_conflict_beats_a_perfect_score() {
        assert_eq!(
            attribution_for(1.0, "project", "tool"),
            Attribution::Distinct
        );
        assert_eq!(
            attribution_for(1.0, "person", "project"),
            Attribution::Distinct
        );
    }

    /// `unknown` asserts nothing, so it must not block a match in either
    /// direction — otherwise the model's uncertainty would permanently
    /// fragment the graph.
    #[test]
    fn unknown_type_matches_anything() {
        assert_eq!(
            attribution_for(1.0, UNKNOWN_ENTITY_TYPE, "project"),
            Attribution::Attribute,
        );
        assert_eq!(
            attribution_for(1.0, "project", UNKNOWN_ENTITY_TYPE),
            Attribution::Attribute,
        );
    }

    // ── Duplicate identity ───────────────────────────────────────────────

    /// Predicate spelling can no longer split a duplicate: the vocabulary is
    /// closed, so `Uses`, `utilises` and `depends_on` all *parse* to the same
    /// variant long before the key is built.
    #[test]
    fn predicate_synonyms_resolve_to_one_variant() {
        for spelling in ["uses", "Uses", "  USES ", "utilises", "depends_on", "used"] {
            assert_eq!(
                Predicate::parse(spelling),
                Some(Predicate::Uses),
                "{spelling:?} should fold into `uses`",
            );
        }
        assert_eq!(Predicate::parse("frobnicates"), None);
    }

    #[test]
    fn duplicate_key_ignores_literal_case() {
        let mut a = memory_with("a", SourceKind::UserStated, 0.5);
        let mut b = memory_with("b", SourceKind::Derived, 0.9);
        a.predicate = Predicate::Uses;
        b.predicate = Predicate::Uses;
        a.object = EntityRef::Literal("Svc-A".to_string());
        b.object = EntityRef::Literal("svc-a".to_string());
        assert_eq!(duplicate_key(&a), duplicate_key(&b));
    }

    #[test]
    fn duplicate_key_separates_scopes_and_entity_from_literal() {
        let base = memory_with("a", SourceKind::UserStated, 0.5);
        let mut other_project = base.clone();
        other_project.project = Some("owner/other".to_string());
        assert_ne!(duplicate_key(&base), duplicate_key(&other_project));

        let mut global = base.clone();
        global.project = None;
        assert_ne!(duplicate_key(&base), duplicate_key(&global));

        // A literal "svc-a" and an entity that happens to be named "svc-a" are
        // different memories; collapsing them would merge a resolved reference
        // into an unresolved string.
        let mut as_entity = base.clone();
        as_entity.object = EntityRef::Entity("svc-a".to_string());
        assert_ne!(duplicate_key(&base), duplicate_key(&as_entity));
    }

    // ── Parser tolerance ─────────────────────────────────────────────────

    #[test]
    fn parses_memories_with_kind_entity_types_and_relation() {
        let response = r#"{"memories": [
            {"subject": "the user", "subject_is_entity": true, "subject_type": "person",
             "predicate": "prefers", "object": "tabs", "object_is_entity": false,
             "object_type": "unknown", "statement": "the user prefers tabs",
             "kind": "preference", "source_kind": "user_stated", "confidence": 0.9,
             "relation": {"type": "supersedes", "target": "old-1"}}
        ]}"#;
        let parsed = parse_ingestion(response).unwrap();
        assert_eq!(parsed.memories.len(), 1);
        let c = &parsed.memories[0];
        assert_eq!(c.kind, "preference");
        assert_eq!(c.subject_type, "person");
        assert!(!c.object_is_entity);
        let rel = c.relation.as_ref().unwrap();
        assert_eq!(rel.kind, "supersedes");
        assert_eq!(rel.target, "old-1");
    }

    #[test]
    fn parses_fenced_json_and_tolerates_a_missing_relation() {
        let response = "Sure! Here you go:\n```json\n{\"memories\": [\
            {\"subject\": \"the user\", \"subject_is_entity\": true, \"predicate\": \"prefers\", \
             \"object\": \"tabs\", \"object_is_entity\": false, \
             \"statement\": \"prefers tabs\", \"kind\": \"preference\", \
             \"source_kind\": \"user_stated\", \"confidence\": 0.8}]}\n```\nHope that helps!";
        let parsed = parse_ingestion(response).unwrap();
        assert_eq!(parsed.memories.len(), 1);
        assert!(parsed.memories[0].relation.is_none());
    }

    /// Small local models emit markdown escapes that are illegal JSON. The
    /// repair pass is what keeps one stray backslash from costing a whole
    /// session's memory.
    #[test]
    fn repairs_invalid_escapes_before_giving_up() {
        let response = r#"{"memories": [{"subject": "the user", "subject_is_entity": true,
            "predicate": "prefers", "object": "snake\_case", "object_is_entity": false,
            "statement": "the user prefers snake\_case names", "kind": "preference",
            "source_kind": "user_stated", "confidence": 0.7}]}"#;
        let parsed = parse_ingestion(response).unwrap();
        assert_eq!(parsed.memories.len(), 1);
        assert!(parsed.memories[0].statement.contains("snake"));
    }

    #[test]
    fn unknown_kind_and_source_kind_fall_back_rather_than_dropping_the_memory() {
        // A memory is too expensive to lose over an unrecognized enum string.
        assert_eq!(
            MemoryKind::parse("nonsense").unwrap_or(MemoryKind::Fact),
            MemoryKind::Fact
        );
        assert_eq!(
            SourceKind::parse("nonsense").unwrap_or(SourceKind::AssistantInferred),
            SourceKind::AssistantInferred,
        );
    }

    #[test]
    fn rejects_output_without_json() {
        assert!(parse_ingestion("I could not extract anything").is_err());
    }

    #[test]
    fn empty_memory_list_parses() {
        assert!(parse_ingestion(r#"{"memories": []}"#)
            .unwrap()
            .memories
            .is_empty());
    }
}
