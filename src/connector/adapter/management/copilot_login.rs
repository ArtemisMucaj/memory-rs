//! GitHub Copilot device-flow login for `serve`.
//!
//! A native app bundles the `memory-rs` binary but does not put it on the
//! user's PATH, so an interactive terminal login isn't runnable. This exposes
//! the OAuth **device flow** over the management API so a GUI can drive it:
//!
//! 1. `POST /api/llm/copilot/login` requests a device code and returns the
//!    `user_code` + `verification_uri` to show the user, then polls GitHub in
//!    the background until the user authorizes (or the code expires).
//! 2. `GET /api/llm/copilot/login` reports the current [`LoginStatus`] so the UI
//!    can advance from *pending* to *authorized* / *failed*.
//!
//! The device flow and its status machine come from
//! [`gh_copilot_rs::LoginSession`]; this wrapper adds the persistence step —
//! writing the `ghu_…` token into `config.json` on success, so every other
//! Copilot path (models, chat) picks it up.

use std::sync::Arc;

use gh_copilot_rs::{GitHubDeviceFlow, LoginSession, LoginStatus};
use tracing::warn;

use crate::connector::adapter::{CopilotConfig, MemoryConfig};
use crate::domain::DomainError;

/// Shared Copilot-login state for serve mode, wrapping a [`LoginSession`] and
/// persisting the token to `config.json` once the session reports `authorized`.
pub struct CopilotLoginService {
    data_dir: String,
    session: Arc<LoginSession>,
}

impl CopilotLoginService {
    pub fn new(data_dir: String) -> Arc<Self> {
        // A failed device-flow client build leaves the session unusable; fall
        // back to a session that simply reports `failed` on start rather than
        // panicking at construction — `serve` must still boot without it.
        let flow = GitHubDeviceFlow::new()
            .map(|f| Arc::new(f) as Arc<dyn gh_copilot_rs::DeviceFlow>)
            .unwrap_or_else(|e| {
                warn!("copilot login: device-flow client unavailable: {e}");
                Arc::new(UnavailableFlow(e.to_string()))
            });
        Arc::new(Self {
            data_dir,
            session: LoginSession::new(flow),
        })
    }

    /// The current status, for `GET /api/llm/copilot/login`.
    pub async fn status(&self) -> LoginStatus {
        self.session.status().await
    }

    /// Start (or restart) the device flow. Returns the initial status
    /// (`Pending` with the code to display, or `Failed`) immediately; a
    /// background task waits for the session to authorize and persists the
    /// token.
    pub async fn start(self: &Arc<Self>) -> LoginStatus {
        let status = self.session.start().await;

        if matches!(status, LoginStatus::Pending { .. }) {
            let service = Arc::clone(self);
            tokio::spawn(async move {
                service.persist_on_success().await;
            });
        }

        status
    }

    /// Poll the session until it leaves `pending`; on `authorized`, read the
    /// token and write it to `config.json`.
    async fn persist_on_success(&self) {
        loop {
            match self.session.status().await {
                LoginStatus::Pending { .. } => {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
                LoginStatus::Authorized => {
                    if let Some(token) = self.session.token().await {
                        if let Err(e) = self.persist_token(token.expose().to_string()).await {
                            warn!("copilot login: authorized but token not saved: {e}");
                        }
                    }
                    return;
                }
                // Failed or Idle (superseded): nothing to persist.
                _ => return,
            }
        }
    }

    /// Persist the `ghu_…` token into `config.json`'s copilot section,
    /// preserving every other section. Blocking filesystem I/O, so it runs off
    /// the async runtime.
    async fn persist_token(&self, token: String) -> Result<(), DomainError> {
        let data_dir = self.data_dir.clone();
        tokio::task::spawn_blocking(move || -> Result<(), DomainError> {
            let mut config = MemoryConfig::load(&data_dir)?;
            let mut copilot = config.copilot.take().unwrap_or_default();
            copilot.github_token = Some(token);
            config.copilot = Some(copilot);
            config.save(&data_dir)
        })
        .await
        .map_err(|e| DomainError::internal(format!("token write task panicked: {e}")))?
    }

    /// The stored Copilot configuration, or the default when none is saved.
    pub fn config(&self) -> Result<CopilotConfig, DomainError> {
        Ok(MemoryConfig::load(&self.data_dir)?
            .copilot
            .unwrap_or_default())
    }
}

/// Stand-in device flow used when the real client could not be constructed
/// (no TLS backend, say). Every call reports the original error, so the UI shows
/// why login is unavailable instead of a silent no-op.
struct UnavailableFlow(String);

#[async_trait::async_trait]
impl gh_copilot_rs::DeviceFlow for UnavailableFlow {
    async fn request_device_code(
        &self,
    ) -> Result<gh_copilot_rs::DeviceAuthorization, gh_copilot_rs::CopilotError> {
        Err(gh_copilot_rs::CopilotError::transport(self.0.clone()))
    }

    async fn poll_once(
        &self,
        _authorization: &gh_copilot_rs::DeviceAuthorization,
    ) -> Result<gh_copilot_rs::PollOutcome, gh_copilot_rs::CopilotError> {
        Err(gh_copilot_rs::CopilotError::transport(self.0.clone()))
    }
}
