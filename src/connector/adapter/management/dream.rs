//! Dream scheduling for `serve` mode.
//!
//! [`DreamService`] wraps one shared [`MemoryDreamUseCase`] (its internal lock
//! is what serializes scheduled and manually-triggered cycles) together with
//! the resolved [`DreamConfig`], and drives two cadences from a single loop:
//!
//! - every [`SWEEP_INTERVAL_SECS`], a **harvest sweep** imports finished
//!   sessions (idle past the configured window, never imported), so memories
//!   land promptly instead of waiting for the next full dream;
//! - whenever the persisted last-run timestamp says a full cycle is due
//!   (default every 4 h), a **dream cycle** consolidates the store.
//!
//! Scheduling state lives in the memory database (`memory_dream_runs`), so a
//! restarted server continues the cadence instead of dreaming immediately on
//! every boot.
//!
//! The CLI's `dream` command still runs a cycle synchronously in the
//! foreground; this is the background scheduler that only `serve` needs.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use crate::application::use_cases::memory_support::unix_now;
use crate::application::{MemoryDreamUseCase, MemoryRepository};
use crate::connector::adapter::config::{DreamConfig, MemoryConfig};
use crate::connector::api::Container;
use crate::domain::{DomainError, DreamRun};

/// Seconds between scheduler ticks (harvest sweep + dream-due check).
const SWEEP_INTERVAL_SECS: u64 = 15 * 60;

/// Shared dream state for serve mode: the scheduler loop and the management
/// API's status/trigger endpoints both go through this.
pub struct DreamService {
    use_case: Arc<MemoryDreamUseCase>,
    memory_repo: Arc<dyn MemoryRepository>,
    /// The scheduling config, behind a lock so a management-API write applies
    /// live: the scheduler reads a fresh snapshot each tick, so a changed
    /// interval / idle window / toggle takes effect on the next sweep without a
    /// server restart. Guarded by a plain `RwLock` (never held across `.await`).
    config: RwLock<DreamConfig>,
    /// Data dir where `config.json` lives, so config writes can be persisted.
    data_dir: String,
    /// Whether a cycle or sweep is currently in flight (for status reporting;
    /// mutual exclusion itself lives inside the use case).
    running: AtomicBool,
}

impl DreamService {
    /// Build the service from the serve container, using the container's
    /// resolved chat endpoint for all dream model calls and the `dream` section
    /// of `config.json` for scheduling.
    pub fn build(container: &Container) -> Result<Arc<Self>, DomainError> {
        let config = MemoryConfig::load(container.data_dir())?
            .dream
            .unwrap_or_default();
        Ok(Arc::new(Self {
            use_case: Arc::new(container.memory_dream_use_case()?),
            memory_repo: container.memory_repository()?,
            config: RwLock::new(config),
            data_dir: container.data_dir().to_string(),
            running: AtomicBool::new(false),
        }))
    }

    /// A snapshot of the current scheduling config. Cloned so callers never hold
    /// the lock (and never hold a guard across `.await`). A poisoned lock is
    /// logged (not silently swallowed) before falling back to the default.
    pub fn config(&self) -> DreamConfig {
        self.config.read().map(|c| c.clone()).unwrap_or_else(|e| {
            tracing::warn!("dream scheduler config lock poisoned, using default: {e}");
            DreamConfig::default()
        })
    }

    /// Apply new dream settings: persist them into `config.json`'s `dream`
    /// section (preserving every other section) and swap the in-memory config so
    /// the scheduler picks them up on its next tick. Returns the merged config.
    ///
    /// Async because the persistence step does blocking filesystem I/O
    /// (`load` + `save`), which is pushed off the runtime via `spawn_blocking`
    /// so it never stalls the async request thread.
    pub async fn update_config(&self, patch: DreamConfigPatch) -> Result<DreamConfig, DomainError> {
        // Reject nonsensical values up front (durations must be positive), so a
        // `0` is a clear 400 rather than a silently-ignored write — the accessors
        // treat `0` as "use the default", which would mislead the caller.
        patch.validate()?;

        // Merge onto the current in-memory config so an omitted field is left
        // unchanged rather than reset to its default.
        let mut merged = self.config();
        patch.apply(&mut merged);

        // Persist off the async thread: load the whole doc so other sections
        // (openai/embedding) survive the write, replace the dream section, save.
        let data_dir = self.data_dir.clone();
        let to_write = merged.clone();
        tokio::task::spawn_blocking(move || -> Result<(), DomainError> {
            let mut doc = MemoryConfig::load(&data_dir)?;
            doc.dream = Some(to_write);
            doc.save(&data_dir)
        })
        .await
        .map_err(|e| DomainError::internal(format!("config write task panicked: {e}")))??;

        // Swap the live config so the scheduler reads the new values next tick.
        match self.config.write() {
            Ok(mut guard) => *guard = merged.clone(),
            Err(e) => tracing::warn!("failed to swap live dream config (lock poisoned): {e}"),
        }
        Ok(merged)
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    pub async fn last_run(&self) -> Option<DreamRun> {
        match self.memory_repo.last_dream_run().await {
            Ok(run) => run,
            Err(e) => {
                tracing::warn!("failed to read last dream run for status: {e}");
                None
            }
        }
    }

    fn idle_secs(&self) -> i64 {
        (self.config().session_idle_minutes() * 60) as i64
    }

    /// Start a dream cycle in the background. Returns `false` (without
    /// spawning) when one is already in flight.
    pub fn trigger(self: &Arc<Self>) -> bool {
        if self.running.swap(true, Ordering::SeqCst) {
            return false;
        }
        let service = Arc::clone(self);
        tokio::spawn(async move {
            let _reset = RunningGuard(&service.running);
            service.run_cycle().await;
        });
        true
    }

    async fn run_cycle(&self) {
        let auto_import = self.config().auto_import();
        match self.use_case.execute(self.idle_secs(), auto_import).await {
            Ok(report) => tracing::info!(
                "dream cycle finished ({} sessions imported, {} ops applied, {} skipped)",
                report.sessions_imported,
                report.applied.len(),
                report.skipped.len()
            ),
            Err(e) => tracing::warn!("dream cycle failed: {e}"),
        }
    }

    /// Run the scheduler until the process exits.
    ///
    /// The loop always runs so a config change made at runtime (via
    /// `update_config`) takes effect: each tick reads a fresh config snapshot,
    /// so enabling dreaming/auto-import later starts it without a restart. When
    /// both are off, ticks are cheap no-ops.
    pub async fn run_scheduler(self: Arc<Self>) {
        let cfg = self.config();
        tracing::info!(
            "dream scheduler: sweep every {} min, dream every {} h, auto-import {}",
            SWEEP_INTERVAL_SECS / 60,
            cfg.dream_interval_hours(),
            if cfg.auto_import() { "on" } else { "off" },
        );
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(SWEEP_INTERVAL_SECS));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // The first `tick()` completes immediately, so a freshly started server
        // harvests (and dreams, when due) right away.
        loop {
            ticker.tick().await;
            self.tick().await;
        }
    }

    /// One scheduler tick: run a full dream when due, else a harvest sweep.
    async fn tick(&self) {
        let cfg = self.config();
        if cfg.dream_enabled() && self.dream_due().await {
            if self.running.swap(true, Ordering::SeqCst) {
                return; // a manual trigger is in flight; try again next tick
            }
            let _reset = RunningGuard(&self.running);
            self.run_cycle().await;
            return;
        }
        if !cfg.auto_import() {
            return;
        }
        if self.running.swap(true, Ordering::SeqCst) {
            return;
        }
        let _reset = RunningGuard(&self.running);
        match self.use_case.harvest(self.idle_secs()).await {
            Ok(report) if report.sessions_imported > 0 => tracing::info!(
                "dream sweep: imported {} finished session(s)",
                report.sessions_imported
            ),
            Ok(_) => {}
            Err(e) => tracing::warn!("dream sweep failed: {e}"),
        }
    }

    /// A full cycle is due when none was ever recorded or the last one
    /// finished more than the configured interval ago.
    async fn dream_due(&self) -> bool {
        let interval_secs = (self.config().dream_interval_hours() * 3_600) as i64;
        match self.memory_repo.last_dream_run().await {
            Ok(Some(last)) => unix_now() - last.finished_at >= interval_secs,
            Ok(None) => true,
            Err(e) => {
                tracing::warn!("dream scheduler could not read last run: {e}");
                false
            }
        }
    }
}

/// A partial update to the dream scheduling config. Every field is optional so
/// a client can change one setting without resending the rest; an omitted field
/// leaves the current value untouched. A `0` duration is rejected by
/// [`validate`](Self::validate) — the accessors treat `0` as "use the default",
/// so accepting it would silently ignore the client's value.
#[derive(Debug, Default, Clone, serde::Deserialize)]
pub struct DreamConfigPatch {
    pub dream_enabled: Option<bool>,
    pub dream_interval_hours: Option<u64>,
    pub session_idle_minutes: Option<u64>,
    pub auto_import: Option<bool>,
}

impl DreamConfigPatch {
    /// Reject values the scheduler cannot honor. Durations must be positive:
    /// `0` would be treated as "use the default" by the accessors, so accepting
    /// it would silently ignore the client's intent — return a clear error
    /// instead (surfaced as a 400 by the handler).
    fn validate(&self) -> Result<(), DomainError> {
        if self.dream_interval_hours == Some(0) {
            return Err(DomainError::invalid_input(
                "dream_interval_hours must be at least 1",
            ));
        }
        if self.session_idle_minutes == Some(0) {
            return Err(DomainError::invalid_input(
                "session_idle_minutes must be at least 1",
            ));
        }
        Ok(())
    }

    /// Merge this patch onto `config`, overwriting only the fields it sets.
    fn apply(&self, config: &mut DreamConfig) {
        if let Some(v) = self.dream_enabled {
            config.dream_enabled = Some(v);
        }
        if let Some(v) = self.dream_interval_hours {
            config.dream_interval_hours = Some(v);
        }
        if let Some(v) = self.session_idle_minutes {
            config.session_idle_minutes = Some(v);
        }
        if let Some(v) = self.auto_import {
            config.auto_import = Some(v);
        }
    }
}

/// Resets the shared `running` flag when dropped, so a panicking cycle or
/// sweep can never leave the scheduler wedged with `running = true`.
struct RunningGuard<'a>(&'a AtomicBool);

impl Drop for RunningGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_rejects_zero_durations() {
        let patch = DreamConfigPatch {
            dream_interval_hours: Some(0),
            ..Default::default()
        };
        assert!(patch.validate().is_err());

        let patch = DreamConfigPatch {
            session_idle_minutes: Some(0),
            ..Default::default()
        };
        assert!(patch.validate().is_err());
    }

    #[test]
    fn patch_leaves_omitted_fields_untouched() {
        let mut config = DreamConfig {
            session_idle_minutes: Some(30),
            dream_enabled: Some(false),
            dream_interval_hours: Some(8),
            auto_import: Some(false),
        };
        let patch = DreamConfigPatch {
            dream_enabled: Some(true),
            ..Default::default()
        };
        patch.apply(&mut config);

        assert_eq!(config.dream_enabled, Some(true));
        // Everything else survives the partial update.
        assert_eq!(config.session_idle_minutes, Some(30));
        assert_eq!(config.dream_interval_hours, Some(8));
        assert_eq!(config.auto_import, Some(false));
    }

    #[test]
    fn accessors_treat_zero_as_unset() {
        let config = DreamConfig {
            session_idle_minutes: Some(0),
            dream_interval_hours: Some(0),
            ..Default::default()
        };
        assert_eq!(
            config.session_idle_minutes(),
            DreamConfig::DEFAULT_SESSION_IDLE_MINUTES
        );
        assert_eq!(
            config.dream_interval_hours(),
            DreamConfig::DEFAULT_DREAM_INTERVAL_HOURS
        );
    }

    #[test]
    fn accessors_default_toggles_on() {
        let config = DreamConfig::default();
        assert!(config.dream_enabled());
        assert!(config.auto_import());
    }
}
