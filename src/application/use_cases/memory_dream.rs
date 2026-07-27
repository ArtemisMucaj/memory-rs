//! Dream — the global consolidation pass over the memory store.
//!
//! Per-session extraction ([`memory_extraction`](super::memory_extraction))
//! merges new information only into the handful of memories it prefetches, so
//! duplicates, contradictions, and cross-session patterns accumulate between
//! items that were never in the same extraction context. A dream cycle is the
//! global pass that cleans this up, in five phases:
//!
//! 1. **Harvest** — discover finished sessions (idle for at least an hour)
//!    that were never imported, and run them through the import pipeline.
//!    Skipped when auto-import is off, so `auto_import: false` imports no
//!    sessions even while dreaming stays enabled.
//! 2. **Consolidate** — cluster near-duplicate items by embedding similarity,
//!    then let the model merge each cluster. Contradictions are the priority:
//!    conflicting memories are rewritten into one item carrying the boundary
//!    insight (under which conditions each side holds) instead of dropping a
//!    side.
//! 3. **Reflect** — one pass over the whole store proposing a few higher-level
//!    items: repeated experiences promoted to a skill, per-project facts
//!    generalized to global.
//! 4. **Synthesize skills** — a focused pass over the `experience`/`skill`
//!    items, distilling procedures that recur across sessions into reusable
//!    `skill` items (steps, prerequisites, failure modes).
//! 5. **Refresh** — regenerate the whole-memory digest and record the run.
//!
//! Guardrails keep a misbehaving model from wrecking the store: operations are
//! capped per run, consolidation may only delete items belonging to the
//! cluster it was shown, reflection may not delete at all, and total deletions
//! are bounded by a fraction of the store.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde::Deserialize;
use tracing::{debug, info, warn};

use openai_rs::ChatClient;

use crate::application::interfaces::{Embedder, MemoryRepository, SessionDiscovery};
use crate::application::use_cases::import_session::{ImportOutcome, ImportSessionUseCase};
use crate::application::use_cases::memory_dream_prompt as prompt;
use crate::application::use_cases::memory_extraction::{
    extract_json_object, normalize_name, repair_json_string_escapes,
};
use crate::application::use_cases::memory_summary::SummarizeMemoryUseCase;
use crate::application::use_cases::memory_support::{unix_now, upsert_preserving_identity};
use crate::domain::{
    cosine_similarity, DomainError, DreamRun, ImportedSession, MemoryItem, MemoryKind,
    MemoryOperation, SessionStatus,
};

/// Default idle time after which a discovered session counts as finished.
pub const DEFAULT_SESSION_IDLE_SECS: i64 = 3_600;

/// Cosine similarity above which two items are considered the same topic and
/// clustered for consolidation.
const SIMILARITY_THRESHOLD: f32 = 0.82;

/// Most clusters examined per cycle (largest first); the rest wait for the
/// next dream, keeping a cycle's LLM cost bounded.
const MAX_CLUSTERS_PER_RUN: usize = 8;

/// Most items sent to the model per cluster (most recently updated first).
const MAX_CLUSTER_ITEMS: usize = 6;

/// Most sessions imported by one harvest, so a first run over a large backlog
/// does not turn into hundreds of extraction calls. The rest are picked up by
/// subsequent cycles.
const MAX_HARVEST_SESSIONS: usize = 10;

/// Upper bound on operations applied by one dream cycle.
const MAX_DREAM_OPERATIONS: usize = 32;

/// Most items reflection may propose per cycle.
const MAX_REFLECTION_ITEMS: usize = 5;

/// Reflection is skipped below this store size — too little evidence for
/// cross-item patterns to exist.
const MIN_REFLECTION_ITEMS: usize = 4;

/// Most skill items synthesis may propose per cycle.
const MAX_SKILL_SYNTHESIS_ITEMS: usize = 3;

/// Skill synthesis is skipped below this many procedural (`experience`/`skill`)
/// items — a procedure has to recur to be worth distilling.
const MIN_SKILL_SYNTHESIS_ITEMS: usize = 3;

/// Delete budget per cycle: a fifth of the store, with a floor of one so a
/// small store can still merge a duplicate pair. A model gone wrong can
/// therefore never wipe more than a fraction of the store in one run; the
/// rest of a large cleanup waits for later cycles.
const DELETE_CAP_DIVISOR: usize = 5;

/// What one dream cycle did.
#[derive(Debug, Default)]
pub struct DreamReport {
    /// Finished, never-imported sessions found by discovery.
    pub sessions_eligible: usize,
    /// Sessions actually imported this cycle.
    pub sessions_imported: usize,
    /// Sessions whose import failed and were marked so they are not retried.
    pub sessions_failed: usize,
    /// Similarity clusters examined by consolidation.
    pub clusters_found: usize,
    /// Operations applied, in order.
    pub applied: Vec<MemoryOperation>,
    /// Operations rejected by a guardrail, with the reason.
    pub skipped: Vec<(MemoryOperation, String)>,
}

/// Result of a standalone harvest sweep (serve mode runs these between full
/// dream cycles so finished sessions are imported promptly).
#[derive(Debug, Default)]
pub struct HarvestReport {
    pub sessions_eligible: usize,
    pub sessions_imported: usize,
    /// Sessions whose import failed and were marked so they are not retried.
    pub sessions_failed: usize,
}

/// JSON shape the consolidation/reflection model must return.
#[derive(Debug, Deserialize)]
struct DreamOutput {
    #[serde(default)]
    items: Vec<RawDreamItem>,
    #[serde(default)]
    delete: Vec<RawDreamDelete>,
}

#[derive(Debug, Deserialize)]
struct RawDreamItem {
    #[serde(default)]
    kind: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    content: String,
    /// Project project, or `null`/absent for a global item.
    #[serde(default)]
    project: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawDreamDelete {
    #[serde(default)]
    kind: String,
    #[serde(default)]
    name: String,
}

pub struct MemoryDreamUseCase {
    memory_repo: Arc<dyn MemoryRepository>,
    chat_client: Arc<dyn ChatClient>,
    embedding_service: Arc<Embedder>,
    discovery: Arc<dyn SessionDiscovery>,
    import: ImportSessionUseCase,
    summary: SummarizeMemoryUseCase,
    /// Serializes cycles: a scheduled dream and a manual trigger must never
    /// interleave writes. A plain atomic flag (rather than a `MutexGuard`) is
    /// used because the guard is held across the cycle's `.await` points, and
    /// `MutexGuard` must not cross an await. The loser of the CAS fails fast
    /// instead of queueing a redundant second cycle.
    running: AtomicBool,
}

/// RAII guard clearing [`MemoryDreamUseCase::running`] when a cycle ends,
/// including on early return via `?`, so a failed cycle never wedges the flag.
struct RunningGuard<'a>(&'a AtomicBool);

impl Drop for RunningGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl MemoryDreamUseCase {
    pub fn new(
        memory_repo: Arc<dyn MemoryRepository>,
        chat_client: Arc<dyn ChatClient>,
        embedding_service: Arc<Embedder>,
        discovery: Arc<dyn SessionDiscovery>,
        import: ImportSessionUseCase,
        summary: SummarizeMemoryUseCase,
    ) -> Self {
        Self {
            memory_repo,
            chat_client,
            embedding_service,
            discovery,
            import,
            summary,
            running: AtomicBool::new(false),
        }
    }

    /// Acquire the single-cycle guard, failing fast if another cycle is active.
    fn begin_cycle(&self) -> Result<RunningGuard<'_>, DomainError> {
        self.running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| DomainError::invalid_input("a dream cycle is already running"))?;
        Ok(RunningGuard(&self.running))
    }

    /// Run one full dream cycle. `session_idle_secs` is how long a session
    /// must have been inactive to count as finished. `auto_import` gates the
    /// harvest phase: when `false`, the cycle consolidates and reflects over the
    /// existing store but imports no new sessions — so `auto_import: false`
    /// means "never import sessions automatically", dreaming included.
    #[tracing::instrument(skip_all)]
    pub async fn execute(
        &self,
        session_idle_secs: i64,
        auto_import: bool,
    ) -> Result<DreamReport, DomainError> {
        let _guard = self.begin_cycle()?;
        let started_at = unix_now();
        let mut report = DreamReport::default();

        // Any phase can write memory before a later phase errors out, so run
        // the cycle body separately and record the run either way — a failed
        // cycle that already harvested or consolidated must still leave a trace
        // in history with the counts it managed to apply.
        match self
            .run_cycle(session_idle_secs, auto_import, &mut report)
            .await
        {
            Ok(()) => {
                self.record_run(&report, started_at, "completed").await;
                info!(
                    "dream cycle finished: {} imported, {} clusters, {} ops applied, {} skipped",
                    report.sessions_imported,
                    report.clusters_found,
                    report.applied.len(),
                    report.skipped.len()
                );
                Ok(report)
            }
            Err(e) => {
                self.record_run(&report, started_at, &format!("failed: {e}"))
                    .await;
                warn!(
                    "dream cycle failed after {} ops applied, {} imported: {e}",
                    report.applied.len(),
                    report.sessions_imported
                );
                Err(e)
            }
        }
    }

    /// The body of one dream cycle (all five phases), factored out so
    /// [`execute`](Self::execute) can record the run on both success and error.
    async fn run_cycle(
        &self,
        session_idle_secs: i64,
        auto_import: bool,
        report: &mut DreamReport,
    ) -> Result<(), DomainError> {
        // Phase 1 — harvest. Skipped when auto-import is off: dreaming then only
        // consolidates and reflects over already-imported memories, and no new
        // session is pulled in behind the user's back.
        if auto_import {
            let harvest = self.harvest_inner(session_idle_secs).await?;
            report.sessions_eligible = harvest.sessions_eligible;
            report.sessions_imported = harvest.sessions_imported;
            report.sessions_failed = harvest.sessions_failed;
        }

        let items = self.memory_repo.list_items(None).await?;

        // Phase 2 — consolidate near-duplicate clusters.
        let delete_budget = (items.len() / DELETE_CAP_DIVISOR).max(1);
        let mut deletes_used = 0usize;
        let clusters = self.build_clusters(&items).await?;
        report.clusters_found = clusters.len();
        for cluster in clusters {
            let operations = match self.consolidate_cluster(&cluster).await {
                Ok(ops) => ops,
                Err(e) => {
                    warn!("dream consolidation call failed, skipping cluster: {e}");
                    continue;
                }
            };
            // Consolidation may only delete what it was shown.
            let deletable: HashSet<(MemoryKind, String)> = cluster
                .iter()
                .map(|item| (item.kind(), item.name().to_string()))
                .collect();
            self.apply(
                operations,
                Some(&deletable),
                delete_budget,
                &mut deletes_used,
                report,
            )
            .await?;
        }

        // Phase 3 — reflect over the whole store (writes only, no deletes).
        // Reload first: consolidation just rewrote/deleted items, and stale
        // inputs would let reflection resurrect what was merged away.
        let items = self.memory_repo.list_items(None).await?;
        if items.len() >= MIN_REFLECTION_ITEMS {
            match self.reflect(&items).await {
                Ok(operations) => {
                    self.apply(
                        operations,
                        Some(&HashSet::new()), // nothing is deletable
                        delete_budget,
                        &mut deletes_used,
                        report,
                    )
                    .await?;
                }
                Err(e) => warn!("dream reflection call failed, skipping: {e}"),
            }
        }

        // Phase 3.5 — synthesize skills from recurring procedural memory.
        // Reload so freshly reflected items are in view, then look only at the
        // `experience`/`skill` items — the raw material for reusable procedures.
        let items = self.memory_repo.list_items(None).await?;
        let procedural: Vec<MemoryItem> = items
            .into_iter()
            .filter(|item| matches!(item.kind(), MemoryKind::Experience | MemoryKind::Skill))
            .collect();
        if procedural.len() >= MIN_SKILL_SYNTHESIS_ITEMS {
            match self.synthesize_skills(&procedural).await {
                Ok(operations) => {
                    self.apply(
                        operations,
                        Some(&HashSet::new()), // write-only, like reflection
                        delete_budget,
                        &mut deletes_used,
                        report,
                    )
                    .await?;
                }
                Err(e) => warn!("dream skill synthesis call failed, skipping: {e}"),
            }
        }

        // Phase 5 — refresh the digests. A failure here means memory writes
        // landed but their digests are now stale, so it is propagated (not
        // swallowed): the caller then finalizes the run as failed rather than
        // recording a misleading "completed".
        if !report.applied.is_empty() {
            self.summary.regenerate_digest().await?;
        }
        // Per-project digests check their own staleness, so this only spends
        // model calls on projects the cycle (or anything since the last one)
        // actually touched.
        self.summary.regenerate_project_digests().await?;
        Ok(())
    }

    /// Import finished, never-imported sessions (the harvest phase alone).
    /// Serve mode calls this on a short interval between full dream cycles.
    pub async fn harvest(&self, session_idle_secs: i64) -> Result<HarvestReport, DomainError> {
        let _guard = self.begin_cycle()?;
        self.harvest_inner(session_idle_secs).await
    }

    /// Mark a session that could not be imported, so later cycles skip it.
    ///
    /// Without this the session stays unmarked and every scheduled cycle tries
    /// it again forever — an unreadable transcript, or one a small model never
    /// returns valid JSON for, would burn an LLM call on each pass. The row
    /// records why it failed; a deliberate `codesearch memory import <path>
    /// --force` still retries it.
    ///
    /// A failure to *write* the marker is only logged: it must not abort the
    /// harvest, and the worst case is the session being retried next cycle,
    /// which is the old behaviour.
    async fn record_failed_harvest(&self, id: &str, source: &str, error: &str) {
        let session = ImportedSession {
            id: id.to_string(),
            source: source.to_string(),
            imported_at: unix_now(),
            message_count: 0,
            items_written: 0,
            status: SessionStatus::Failed,
            last_error: Some(error.to_string()),
        };
        if let Err(e) = self.memory_repo.record_session(&session).await {
            warn!("dream harvest: could not record failure for '{id}': {e}");
        }
    }

    async fn harvest_inner(&self, session_idle_secs: i64) -> Result<HarvestReport, DomainError> {
        let mut report = HarvestReport::default();
        let sessions = self.discovery.discover().await?;
        let imported: HashSet<String> = self
            .memory_repo
            .list_sessions()
            .await?
            .into_iter()
            .map(|s| s.id)
            .collect();
        let now = unix_now();
        let mut attempted = 0usize;

        for session in sessions {
            if session.updated_at <= 0 || now - session.updated_at < session_idle_secs {
                continue;
            }
            if imported.contains(&session.id) {
                continue;
            }
            report.sessions_eligible += 1;
            // Every *attempt* counts against the budget, not just the ones that
            // succeed. Gating on successes alone let a batch of failing sessions
            // run unbounded, since a failure never moved the counter.
            if attempted >= MAX_HARVEST_SESSIONS {
                continue;
            }
            attempted += 1;
            let transcript = match self.discovery.load_transcript(&session).await {
                Ok(t) => t,
                Err(e) => {
                    warn!(
                        "dream harvest: could not load session '{}': {e}",
                        session.id
                    );
                    // The transcript never loaded, so its own `source` string is
                    // unavailable; rebuild the discovery form (`claude:<id>`).
                    let source = format!("{}:{}", session.source.as_str(), session.id);
                    self.record_failed_harvest(&session.id, &source, &e.to_string())
                        .await;
                    report.sessions_failed += 1;
                    continue;
                }
            };
            match self.import.execute(&transcript, false).await {
                Ok(ImportOutcome::Imported { session, .. }) => {
                    info!("dream harvest: imported session '{}'", session.id);
                    report.sessions_imported += 1;
                }
                Ok(ImportOutcome::AlreadyImported { .. }) => {}
                Err(e) => {
                    warn!("dream harvest: import of '{}' failed: {e}", session.id);
                    self.record_failed_harvest(&transcript.id, &transcript.source, &e.to_string())
                        .await;
                    report.sessions_failed += 1;
                }
            }
        }
        Ok(report)
    }

    /// Group items into similarity clusters (connected components over pairs
    /// whose embedding cosine similarity crosses the threshold). Items without
    /// a stored vector cannot be clustered and are left alone.
    async fn build_clusters(
        &self,
        items: &[MemoryItem],
    ) -> Result<Vec<Vec<MemoryItem>>, DomainError> {
        let vectors = self.memory_repo.list_item_vectors().await?;
        let by_id: std::collections::HashMap<&str, &MemoryItem> =
            items.iter().map(|item| (item.id(), item)).collect();
        // Keep only vectors whose item still exists, in a stable order.
        let embedded: Vec<(&MemoryItem, &Vec<f32>)> = vectors
            .iter()
            .filter_map(|(id, vector)| by_id.get(id.as_str()).map(|item| (*item, vector)))
            .collect();

        let mut parent: Vec<usize> = (0..embedded.len()).collect();
        for a in 0..embedded.len() {
            for b in (a + 1)..embedded.len() {
                if cosine_similarity(embedded[a].1, embedded[b].1) >= SIMILARITY_THRESHOLD {
                    union(&mut parent, a, b);
                }
            }
        }

        // Group indices first; only the items surviving the caps get cloned.
        let mut groups: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for idx in 0..embedded.len() {
            groups.entry(find(&mut parent, idx)).or_default().push(idx);
        }

        let mut clusters: Vec<Vec<usize>> = groups
            .into_values()
            .filter(|group| group.len() >= 2)
            .collect();
        for cluster in &mut clusters {
            // Most recently updated first; the model sees the freshest take at
            // the top and the prompt truncation drops the stalest.
            cluster.sort_by(|&a, &b| {
                embedded[b]
                    .0
                    .updated_at()
                    .cmp(&embedded[a].0.updated_at())
                    .then_with(|| embedded[a].0.name().cmp(embedded[b].0.name()))
            });
            cluster.truncate(MAX_CLUSTER_ITEMS);
        }
        // Largest (most redundant) clusters first; a deterministic tiebreak
        // keeps runs reproducible.
        clusters.sort_by(|a, b| {
            b.len()
                .cmp(&a.len())
                .then_with(|| embedded[a[0]].0.name().cmp(embedded[b[0]].0.name()))
        });
        clusters.truncate(MAX_CLUSTERS_PER_RUN);
        debug!("dream: {} consolidation clusters", clusters.len());
        Ok(clusters
            .into_iter()
            .map(|group| {
                group
                    .into_iter()
                    .map(|idx| embedded[idx].0.clone())
                    .collect()
            })
            .collect())
    }

    /// One consolidation call for one cluster, with a format-recovery retry.
    async fn consolidate_cluster(
        &self,
        cluster: &[MemoryItem],
    ) -> Result<Vec<MemoryOperation>, DomainError> {
        let system = prompt::consolidation_system_prompt();
        let user = prompt::consolidation_user_prompt(cluster);
        self.complete_operations(&system, &user).await
    }

    /// One reflection call over the whole store, with a format-recovery retry.
    /// Proposed items are capped; deletes are stripped by the caller.
    async fn reflect(&self, items: &[MemoryItem]) -> Result<Vec<MemoryOperation>, DomainError> {
        let system = prompt::reflection_system_prompt(MAX_REFLECTION_ITEMS);
        let user = prompt::reflection_user_prompt(items);
        let mut operations = self.complete_operations(&system, &user).await?;
        cap_upserts(&mut operations, MAX_REFLECTION_ITEMS, None);
        Ok(operations)
    }

    /// One skill-synthesis call over the store's procedural (`experience`/
    /// `skill`) items, with a format-recovery retry. Only `skill` upserts are
    /// kept (the prompt asks for skills; a stray other-kind item is dropped),
    /// capped to [`MAX_SKILL_SYNTHESIS_ITEMS`]; deletes are stripped by the
    /// caller.
    async fn synthesize_skills(
        &self,
        items: &[MemoryItem],
    ) -> Result<Vec<MemoryOperation>, DomainError> {
        let system = prompt::skill_synthesis_system_prompt(MAX_SKILL_SYNTHESIS_ITEMS);
        let refs: Vec<&MemoryItem> = items.iter().collect();
        let user = prompt::skill_synthesis_user_prompt(&refs);
        let mut operations = self.complete_operations(&system, &user).await?;
        cap_upserts(
            &mut operations,
            MAX_SKILL_SYNTHESIS_ITEMS,
            Some(MemoryKind::Skill),
        );
        Ok(operations)
    }

    /// Send one dream prompt and parse its operations, retrying once with a
    /// format-correction message when the output is unparseable.
    async fn complete_operations(
        &self,
        system: &str,
        user: &str,
    ) -> Result<Vec<MemoryOperation>, DomainError> {
        let schema = prompt::dream_schema();
        let response = self
            .chat_client
            .complete_json(system, user, "memory_dream", &schema)
            .await?;
        match parse_dream_operations(&response) {
            Ok(ops) => Ok(ops),
            Err(first_err) => {
                debug!("dream output unparseable, retrying once: {first_err}");
                let retry_user = format!("{user}\n\n{}", prompt::format_retry_prompt());
                let response = self
                    .chat_client
                    .complete_json(system, &retry_user, "memory_dream", &schema)
                    .await?;
                parse_dream_operations(&response).map_err(|e| {
                    DomainError::parse(format!(
                        "dream model returned unparseable output twice: {e}"
                    ))
                })
            }
        }
    }

    /// Apply validated operations under the run-level guardrails. `deletable`
    /// restricts which `(kind, name)` keys may be deleted (`Some(empty)`
    /// forbids deletion outright).
    async fn apply(
        &self,
        operations: Vec<MemoryOperation>,
        deletable: Option<&HashSet<(MemoryKind, String)>>,
        delete_budget: usize,
        deletes_used: &mut usize,
        report: &mut DreamReport,
    ) -> Result<(), DomainError> {
        // Names upserted this cycle must not be deleted by a later operation
        // of the same cycle (a model merging A+B into A sometimes also lists A
        // for deletion).
        let mut upserted: HashSet<(MemoryKind, String)> = report
            .applied
            .iter()
            .filter_map(|op| match op {
                MemoryOperation::Upsert { kind, name, .. } => Some((*kind, name.clone())),
                MemoryOperation::Delete { .. } => None,
            })
            .collect();

        for op in operations {
            if report.applied.len() >= MAX_DREAM_OPERATIONS {
                report
                    .skipped
                    .push((op, "operation limit reached".to_string()));
                continue;
            }
            match op {
                MemoryOperation::Upsert { kind, ref name, .. } => {
                    upserted.insert((kind, name.clone()));
                    self.apply_upsert(&op).await?;
                    report.applied.push(op);
                }
                MemoryOperation::Delete { kind, ref name } => {
                    let key = (kind, name.clone());
                    if upserted.contains(&key) {
                        report
                            .skipped
                            .push((op, "name was upserted this cycle".to_string()));
                        continue;
                    }
                    if let Some(allowed) = deletable {
                        if !allowed.contains(&key) {
                            report.skipped.push((
                                op,
                                "delete target was not part of the examined cluster".to_string(),
                            ));
                            continue;
                        }
                    }
                    if *deletes_used >= delete_budget {
                        report
                            .skipped
                            .push((op, "delete budget for this cycle exhausted".to_string()));
                        continue;
                    }
                    // `(kind, name)` no longer identifies a single item, so a
                    // name living in several projects is ambiguous here — the
                    // cluster key carries no project to choose by. Deleting all
                    // of them would destroy other projects' memories, so an
                    // ambiguous target is left alone.
                    let candidates = self.memory_repo.find_items_named(kind, name).await?;
                    match candidates.as_slice() {
                        [item] => {
                            self.memory_repo.delete_item_by_id(item.id()).await?;
                            *deletes_used += 1;
                            report.applied.push(op);
                        }
                        [] => report.skipped.push((op, "item not found".to_string())),
                        _ => report.skipped.push((
                            op,
                            "name is ambiguous across projects; not deleting".to_string(),
                        )),
                    }
                }
            }
        }
        Ok(())
    }

    /// Write one upsert through the shared identity-preserving path (dream
    /// keeps the existing item's source session, so no override).
    async fn apply_upsert(&self, op: &MemoryOperation) -> Result<(), DomainError> {
        let MemoryOperation::Upsert {
            kind,
            name,
            content,
            project,
        } = op
        else {
            return Ok(());
        };
        upsert_preserving_identity(
            self.memory_repo.as_ref(),
            self.embedding_service.as_ref(),
            *kind,
            name,
            content,
            project.clone(),
            None,
            unix_now(),
        )
        .await
    }

    /// Best-effort persistence of the run record (a bookkeeping failure must
    /// not fail a cycle whose memory writes already succeeded). `status` is
    /// `"completed"` or `"failed: <reason>"`, carrying the counts applied so
    /// far so a partial run is still inspectable.
    async fn record_run(&self, report: &DreamReport, started_at: i64, status: &str) {
        let run = DreamRun {
            id: uuid::Uuid::new_v4().to_string(),
            started_at,
            finished_at: unix_now(),
            sessions_imported: report.sessions_imported,
            clusters_found: report.clusters_found,
            operations_applied: report.applied.len(),
            operations_skipped: report.skipped.len(),
            status: status.to_string(),
        };
        if let Err(e) = self.memory_repo.record_dream_run(&run).await {
            warn!("failed to record dream run: {e}");
        }
    }
}

/// Cap a write-only pass's proposed upserts: keep at most `max_upserts`
/// upserts, dropping any whose kind does not match `required_kind` (when set).
/// Deletes are left in place — the caller rejects them with a reason so the
/// skip is recorded.
fn cap_upserts(
    operations: &mut Vec<MemoryOperation>,
    max_upserts: usize,
    required_kind: Option<MemoryKind>,
) {
    let mut kept = 0usize;
    operations.retain(|op| match op {
        MemoryOperation::Upsert { kind, .. } => {
            if required_kind.is_some_and(|required| *kind != required) {
                return false;
            }
            kept += 1;
            kept <= max_upserts
        }
        MemoryOperation::Delete { .. } => true,
    });
}

/// Parse the model's dream JSON into validated, normalized operations,
/// tolerating prose/fences and the invalid-escape output of small models.
fn parse_dream_operations(response: &str) -> Result<Vec<MemoryOperation>, DomainError> {
    let json = extract_json_object(response)
        .ok_or_else(|| DomainError::parse("no JSON object found in dream output"))?;
    let output: DreamOutput = match serde_json::from_str(json) {
        Ok(output) => output,
        Err(strict_err) => {
            let repaired = repair_json_string_escapes(json);
            serde_json::from_str(&repaired)
                .map_err(|_| DomainError::parse(format!("invalid dream JSON: {strict_err}")))?
        }
    };

    let mut operations = Vec::new();
    for item in output.items {
        let Some(kind) = MemoryKind::parse(&item.kind) else {
            continue;
        };
        let Some(name) = normalize_name(&item.name) else {
            continue;
        };
        let content = item.content.trim();
        if content.is_empty() {
            continue;
        }
        let project = item
            .project
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty() && !s.eq_ignore_ascii_case("null"))
            .map(str::to_string);
        operations.push(MemoryOperation::Upsert {
            kind,
            name,
            content: content.to_string(),
            project,
        });
    }
    for del in output.delete {
        let Some(kind) = MemoryKind::parse(&del.kind) else {
            continue;
        };
        let Some(name) = normalize_name(&del.name) else {
            continue;
        };
        operations.push(MemoryOperation::Delete { kind, name });
    }
    Ok(operations)
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        parent[rb] = ra;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_dream_items_and_deletes() {
        let response = r#"{"items": [
            {"kind": "experience", "name": "Duckdb Locking", "content": "conflicts on concurrent writers", "project": null},
            {"kind": "fact", "name": "sdk_version", "content": "pinned to 2.1", "project": "svc-a"}
        ], "delete": [{"kind": "fact", "name": "old_take"}]}"#;
        let ops = parse_dream_operations(response).unwrap();
        assert_eq!(ops.len(), 3);
        assert_eq!(
            ops[0],
            MemoryOperation::Upsert {
                kind: MemoryKind::Experience,
                name: "duckdb_locking".to_string(),
                content: "conflicts on concurrent writers".to_string(),
                project: None,
            }
        );
        let MemoryOperation::Upsert { project, .. } = &ops[1] else {
            panic!("expected upsert");
        };
        assert_eq!(project.as_deref(), Some("svc-a"));
        assert_eq!(
            ops[2],
            MemoryOperation::Delete {
                kind: MemoryKind::Fact,
                name: "old_take".to_string(),
            }
        );
    }

    #[test]
    fn dream_parse_skips_unknown_kinds_and_empty_content() {
        let response = r#"{"items": [
            {"kind": "opinion", "name": "x", "content": "y", "project": null},
            {"kind": "fact", "name": "ok", "content": "  ", "project": null}
        ], "delete": [{"kind": "nope", "name": "x"}]}"#;
        let ops = parse_dream_operations(response).unwrap();
        assert!(ops.is_empty());
    }

    #[test]
    fn dream_parse_treats_string_null_project_as_global() {
        let response = r#"{"items": [
            {"kind": "fact", "name": "n", "content": "c", "project": "null"}
        ], "delete": []}"#;
        let ops = parse_dream_operations(response).unwrap();
        let MemoryOperation::Upsert { project, .. } = &ops[0] else {
            panic!("expected upsert");
        };
        assert_eq!(*project, None);
    }

    #[test]
    fn union_find_groups_transitively() {
        let mut parent: Vec<usize> = (0..4).collect();
        union(&mut parent, 0, 1);
        union(&mut parent, 1, 2);
        assert_eq!(find(&mut parent, 2), find(&mut parent, 0));
        assert_ne!(find(&mut parent, 3), find(&mut parent, 0));
    }
}
