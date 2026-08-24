//! Dream — the scheduled harvest of finished sessions.
//!
//! A dream cycle is a single phase: discover finished sessions that were
//! never imported and run them through the import pipeline. Only namespaced
//! projects are eligible, and only for sessions newer than the namespace's
//! creation date, so a first harvest does not import years of unrelated
//! history. Skipped entirely when `auto_import` is off.
//!
//! Consolidation, reflection and skill synthesis are gone: they existed to
//! maintain typed edges between memories and to merge near-duplicate items
//! under the old `MemoryItem` model. With edges removed and updates handled
//! as hard delete + insert on the ingestion path, there is nothing left for
//! an offline pass to reorganize.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tracing::{debug, info, warn};

use crate::application::interfaces::{MemoryRepository, SessionDiscovery};
use crate::application::use_cases::import_session::{ImportOutcome, ImportSessionUseCase};
use crate::application::use_cases::llm_json::unix_now;
use crate::domain::{DomainError, ImportedSession, SessionStatus};

/// Default idle time after which a discovered session counts as finished.
pub const DEFAULT_SESSION_IDLE_SECS: i64 = 3_600;

/// Most sessions imported by one harvest, so a first run over a large backlog
/// does not turn into hundreds of extraction calls. The rest are picked up by
/// subsequent cycles.
const MAX_HARVEST_SESSIONS: usize = 10;

/// Whether a session is auto-import eligible against its namespace's creation
/// date. Strict: both sides are whole seconds, so a tie says nothing about the
/// real order and is treated as predating the opt-in.
fn is_after_cutoff(updated_at: i64, cutoff: i64) -> bool {
    updated_at > cutoff
}

/// What one harvest sweep did.
#[derive(Debug, Default)]
pub struct HarvestReport {
    /// Finished, never-imported sessions found by discovery.
    pub sessions_eligible: usize,
    /// Sessions actually imported this cycle.
    pub sessions_imported: usize,
    /// Sessions whose import failed and were marked so they are not retried.
    pub sessions_failed: usize,
}

/// Alias kept for the serve surface: one dream cycle is now exactly one
/// harvest. The old consolidation phases are gone, so the report is the
/// harvest report.
pub type DreamReport = HarvestReport;

/// The dream use case is now just a harvest; the struct keeps its old name
/// because the CLI and serve surfaces already reference it.
pub struct MemoryDreamUseCase {
    memory_repo: Arc<dyn MemoryRepository>,
    discovery: Arc<dyn SessionDiscovery>,
    import: ImportSessionUseCase,
    /// Serializes cycles: a scheduled harvest and a manual trigger must never
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
        discovery: Arc<dyn SessionDiscovery>,
        import: ImportSessionUseCase,
    ) -> Self {
        Self {
            memory_repo,
            discovery,
            import,
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

    /// Run one harvest sweep: import finished, never-imported sessions.
    /// `session_idle_secs` is how long a session must have been inactive to
    /// count as finished.
    #[tracing::instrument(skip_all)]
    pub async fn harvest(&self, session_idle_secs: i64) -> Result<HarvestReport, DomainError> {
        let _guard = self.begin_cycle()?;
        let mut report = HarvestReport::default();
        let sessions = self.discovery.discover().await?;
        let imported: HashSet<String> = self
            .memory_repo
            .list_sessions(None, usize::MAX)
            .await?
            .into_iter()
            .map(|s| s.id)
            .collect();
        // Namespacing a project is the auto-import opt-in; the namespace's
        // creation date keeps it forward-looking. Without this, a first
        // harvest on a machine with years of history would import thousands of
        // unrelated sessions. Manual `memory-rs import` bypasses all of it.
        let namespaces = self.memory_repo.list_namespaces().await?;
        let mut cutoffs: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        for (namespace, _count) in namespaces {
            let Some(created_at) = self.memory_repo.namespace_created_at(&namespace).await? else {
                continue;
            };
            for project in self.memory_repo.namespace_projects(&namespace).await? {
                // A project in several namespaces uses the oldest cutoff, so
                // joining a second namespace can only widen eligibility, never
                // silently narrow it.
                cutoffs
                    .entry(project)
                    .and_modify(|c| *c = (*c).min(created_at))
                    .or_insert(created_at);
            }
        }
        if cutoffs.is_empty() {
            debug!("dream harvest: no namespaced projects, nothing is auto-importable");
            return Ok(report);
        }
        let now = unix_now();
        let mut attempted = 0usize;

        for session in sessions {
            if session.updated_at <= 0 || now - session.updated_at < session_idle_secs {
                continue;
            }
            if imported.contains(&session.id) {
                continue;
            }
            // Checked before loading the transcript, so an out-of-scope
            // session costs nothing beyond the cwd lookup.
            match self
                .discovery
                .session_project(&session)
                .and_then(|project| cutoffs.get(&project).copied())
            {
                Some(cutoff) if is_after_cutoff(session.updated_at, cutoff) => {}
                Some(_) => {
                    debug!(
                        "dream harvest: skipping '{}' — predates its namespace",
                        session.id
                    );
                    continue;
                }
                None => continue,
            }
            report.sessions_eligible += 1;
            // Every *attempt* counts against the budget, not just the ones
            // that succeed. Gating on successes alone let a batch of failing
            // sessions run unbounded, since a failure never moved the counter.
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
                    // The transcript never loaded, so its own `source` string
                    // is unavailable; rebuild the discovery form
                    // (`claude:<id>`).
                    let source = format!("{}:{}", session.source.as_str(), session.id);
                    self.record_failed_harvest(&session.id, &source, &e.to_string())
                        .await;
                    report.sessions_failed += 1;
                    continue;
                }
            };
            match self.import.execute(&transcript, false).await {
                Ok(ImportOutcome::Imported { .. }) => {
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

    /// Mark a session that could not be imported, so later cycles skip it.
    ///
    /// Without this the session stays unmarked and every scheduled cycle
    /// tries it again forever — an unreadable transcript, or one a small
    /// model never returns valid JSON for, would burn an LLM call on each
    /// pass. The row records why it failed; a deliberate
    /// `memory-rs import <path> --force` still retries it.
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
            // Unknown: the transcript is what carries the project, and this
            // marker exists precisely because it could not be read.
            project: None,
            items_written: 0,
            status: SessionStatus::Failed,
            last_error: Some(error.to_string()),
        };
        if let Err(e) = self.memory_repo.record_session(&session).await {
            warn!("dream harvest: could not record failure for '{id}': {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_namespace_cutoff_excludes_the_boundary_second() {
        let cutoff = 1_700_000_000;
        assert!(!is_after_cutoff(cutoff - 1, cutoff));
        assert!(!is_after_cutoff(cutoff, cutoff));
        assert!(is_after_cutoff(cutoff + 1, cutoff));
    }
}
