//! Dream scheduling for `serve` mode.
//!
//! [`DreamService`] wraps one shared [`MemoryDreamUseCase`] (its internal
//! lock is what serializes scheduled and manually-triggered harvests)
//! together with the resolved [`DreamConfig`], and drives a single cadence:
//!
//! - every [`SWEEP_INTERVAL_SECS`], a **harvest sweep** imports finished
//!   sessions (idle past the configured window, never imported), so memories
//!   land promptly.
//!
//! There is no consolidation cycle anymore — there is nothing left for an
//! offline pass to reorganize. The `dream_interval_hours` config knob still
//! exists for forward-compat but is unused.
//!
//! The CLI's `dream` command still runs a harvest synchronously in the
//! foreground; this is the background scheduler that only `serve` needs.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use crate::application::MemoryDreamUseCase;
use crate::connector::adapter::config::{DreamConfig, MemoryConfig};
use crate::connector::api::Container;
use crate::domain::DomainError;

/// Seconds between scheduler ticks (harvest sweep).
const SWEEP_INTERVAL_SECS: u64 = 15 * 60;

/// Shared dream state for serve mode: the scheduler loop and the management
/// API's status/trigger endpoints both go through this.
pub struct DreamService {
    use_case: Arc<MemoryDreamUseCase>,
    /// The scheduling config, behind a lock so a management-API write applies
    /// live: the scheduler reads a fresh snapshot each tick, so a changed
    /// interval / idle window / toggle takes effect on the next sweep without
    /// a server restart. Guarded by a plain `RwLock` (never held across
    /// `.await`).
    config: RwLock<DreamConfig>,
    /// Data dir where `config.json` lives, so config writes can be persisted.
    data_dir: String,
    /// Whether a sweep is currently in flight (for status reporting; mutual
    /// exclusion itself lives inside the use case).
    running: AtomicBool,
}

impl DreamService {
    /// Build the service from the serve container, using the container's
    /// resolved chat endpoint for all harvest model calls and the `dream`
    /// section of `config.json` for scheduling.
    pub fn build(container: &Container) -> Result<Arc<Self>, DomainError> {
        let config = MemoryConfig::load(container.data_dir())?
            .dream
            .unwrap_or_default();
        Ok(Arc::new(Self {
            use_case: Arc::new(container.memory_dream_use_case()?),
            config: RwLock::new(config),
            data_dir: container.data_dir().to_string(),
            running: AtomicBool::new(false),
        }))
    }

    /// A snapshot of the current scheduling config. Cloned so callers never
    /// hold the lock (and never hold a guard across `.await`).
    pub fn config(&self) -> DreamConfig {
        self.config.read().map(|c| c.clone()).unwrap_or_else(|e| {
            tracing::warn!("dream scheduler config lock poisoned, using default: {e}");
            DreamConfig::default()
        })
    }

    /// Apply new dream settings: persist them into `config.json`'s `dream`
    /// section (preserving every other section) and swap the in-memory
    /// config so the scheduler picks them up on its next tick.
    pub async fn update_config(&self, patch: DreamConfigPatch) -> Result<DreamConfig, DomainError> {
        // Reject nonsensical values up front (durations must be positive).
        patch.validate()?;

        let mut merged = self.config();
        patch.apply(&mut merged);

        let data_dir = self.data_dir.clone();
        let to_write = merged.clone();
        tokio::task::spawn_blocking(move || -> Result<(), DomainError> {
            let mut doc = MemoryConfig::load(&data_dir)?;
            doc.dream = Some(to_write);
            doc.save(&data_dir)
        })
        .await
        .map_err(|e| DomainError::internal(format!("config write task panicked: {e}")))??;

        match self.config.write() {
            Ok(mut guard) => *guard = merged.clone(),
            Err(e) => tracing::warn!("failed to swap live dream config (lock poisoned): {e}"),
        }
        Ok(merged)
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    fn idle_secs(&self) -> i64 {
        (self.config().session_idle_minutes() * 60) as i64
    }

    /// Start a harvest in the background. Returns `false` (without spawning)
    /// when one is already in flight.
    pub fn trigger(self: &Arc<Self>) -> bool {
        if self.running.swap(true, Ordering::SeqCst) {
            return false;
        }
        let service = Arc::clone(self);
        tokio::spawn(async move {
            let _reset = RunningGuard(&service.running);
            service.run_sweep().await;
        });
        true
    }

    async fn run_sweep(&self) {
        match self.use_case.harvest(self.idle_secs()).await {
            Ok(report) => tracing::info!(
                "dream sweep finished ({} sessions imported, {} failed)",
                report.sessions_imported,
                report.sessions_failed
            ),
            Err(e) => tracing::warn!("dream sweep failed: {e}"),
        }
    }

    /// Run the scheduler until the process exits.
    pub async fn run_scheduler(self: Arc<Self>) {
        let cfg = self.config();
        tracing::info!(
            "dream scheduler: sweep every {} min, auto-import {}",
            SWEEP_INTERVAL_SECS / 60,
            if cfg.auto_import() { "on" } else { "off" },
        );
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(SWEEP_INTERVAL_SECS));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            self.tick().await;
        }
    }

    /// One scheduler tick: run a harvest sweep.
    async fn tick(&self) {
        let cfg = self.config();
        if !cfg.dream_enabled() || !cfg.auto_import() {
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
}

/// A partial update to the dream scheduling config. Every field is optional
/// so a client can change one setting without resending the rest.
#[derive(Debug, Default, Clone, serde::Deserialize)]
pub struct DreamConfigPatch {
    pub dream_enabled: Option<bool>,
    pub dream_interval_hours: Option<u64>,
    pub session_idle_minutes: Option<u64>,
    pub auto_import: Option<bool>,
}

impl DreamConfigPatch {
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

/// Resets the shared `running` flag when dropped.
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
        assert_eq!(config.session_idle_minutes, Some(30));
        assert_eq!(config.dream_interval_hours, Some(8));
        assert_eq!(config.auto_import, Some(false));
    }
}
