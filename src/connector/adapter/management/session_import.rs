//! Session discovery + background import for `serve` mode.
//!
//! [`SessionImportService`] is the serve-mode analogue of the TUI's Import
//! screen ([`crate::tui::screens::import`]): it discovers finished assistant
//! sessions, materializes a transcript on demand, and imports a chosen session
//! **in the background** so the HTTP request returns immediately and the import
//! keeps running even if the client navigates away.
//!
//! The capability itself is not new — the TUI has driven
//! [`SessionDiscovery`] in-process since the import screen landed. What lives
//! here is only the HTTP adapter around it, so a native app can drive the same
//! flow without holding a connection open for a whole LLM extraction.
//!
//! - one shared instance lives in [`super::AppState`],
//! - imports run under `tokio::spawn` and report progress into a status map,
//! - the map is keyed by a session's stable identity `(source, id)` so status
//!   survives re-discovery (the list re-sorts newest-first each time).
//!
//! The status map mirrors the TUI's `Status` state machine
//! (`queued → importing → done | failed`, plus `already_imported` for sessions
//! already present in the store when discovery ran), so a client can render the
//! exact same per-row markers the TUI does.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::Serialize;
use tokio::sync::Mutex;

use crate::application::{ImportOutcome, SessionDiscovery};
use crate::connector::api::Container;
use crate::domain::{DiscoveredSession, DomainError, SessionLocator, SessionSource};

/// Stable identity of a discovered session: `(source, id)`. Used as the status
/// map key so a session's import status follows it across re-discovery.
type SessionKey = (String, String);

/// Import lifecycle of one session, mirroring the TUI's `Status`.
///
/// Serialized in `snake_case` (`already_imported`, …) as the `status` field of
/// each entry in `GET /api/sessions/import`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImportStatus {
    /// Already present in the memory store when discovery ran.
    AlreadyImported,
    /// Accepted by `POST /api/sessions/import`, worker not yet started.
    Queued,
    /// Extraction in progress.
    Importing,
    /// Extraction finished (freshly imported or re-imported).
    Done,
    /// Extraction failed; carries a short reason.
    Failed,
}

/// A status-map entry: the current lifecycle state plus, on terminal states, a
/// one-line summary (Done) or error (Failed) for the client to surface.
#[derive(Debug, Clone, Serialize)]
pub struct StatusEntry {
    pub source: String,
    pub id: String,
    pub status: ImportStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Shared session-import state for serve mode. Discovery is stateless; the
/// status map is the only mutable state and is guarded by an async mutex (held
/// only for the brief map updates, never across an import).
pub struct SessionImportService {
    container: Arc<Container>,
    discovery: Arc<dyn SessionDiscovery>,
    status: Arc<Mutex<HashMap<SessionKey, StatusEntry>>>,
}

impl SessionImportService {
    /// Build the service from the serve container. Cheap — no LLM client is
    /// constructed until an import actually runs.
    pub fn build(container: Arc<Container>) -> Arc<Self> {
        let discovery = container.session_discovery();
        Arc::new(Self {
            container,
            discovery,
            status: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Discover all finished sessions (newest first). Seeds the status map with
    /// `already_imported` for any session already in the store, so the very
    /// first `discover` call — before any `import` — already carries the ✓
    /// markers the TUI shows on open.
    pub async fn discover(&self) -> Result<Vec<DiscoveredSession>, DomainError> {
        let sessions = self.discovery.discover().await?;
        // Cross-reference the store's imported-session records so the client can
        // render ✓ without a second round-trip. Key the set by a normalized
        // `(source, id)` so two sources reusing the same session id can't mark
        // the wrong one as already-imported (the stored `source` is
        // heterogeneous — a `"zed:…"` tag, an `"opencode:…"` tag, or a Claude
        // file path — so it is normalized back to the source tag).
        let repo = self.container.node_repository()?;
        let imported: HashSet<SessionKey> = repo
            .list_sessions()
            .await?
            .into_iter()
            .map(|s| (normalize_source_tag(&s.source), s.id))
            .collect();

        let mut map = self.status.lock().await;
        for s in &sessions {
            let key = session_key(s);
            if imported.contains(&key) {
                // Don't clobber an in-flight/finished import status.
                map.entry(key).or_insert_with(|| StatusEntry {
                    source: s.source.as_str().to_string(),
                    id: s.id.clone(),
                    status: ImportStatus::AlreadyImported,
                    detail: None,
                });
            }
        }
        Ok(sessions)
    }

    /// Materialize the transcript for the session identified by `(source, id)`.
    /// Re-discovers to resolve the opaque [`SessionLocator`] rather than
    /// trusting a client-supplied path.
    pub async fn transcript(
        &self,
        source: &str,
        id: &str,
    ) -> Result<crate::domain::SessionTranscript, DomainError> {
        let session = self.find(source, id).await?;
        self.discovery.load_transcript(&session).await
    }

    /// Queue a background import of the session identified by `(source, id)`.
    ///
    /// Returns immediately after setting the status to `queued` and spawning the
    /// worker; the import (transcript load → memory extraction → summarization)
    /// runs on a detached task, so it survives the HTTP request completing and
    /// the client navigating away. Re-importing a done/already-imported session
    /// is allowed (extraction is forced); a session already `queued`/`importing`
    /// is a no-op so a double click can't double-run.
    pub async fn import(
        self: &Arc<Self>,
        source: &str,
        id: &str,
        force: bool,
    ) -> Result<(), DomainError> {
        let session = self.find(source, id).await?;
        let key = session_key(&session);

        {
            let mut map = self.status.lock().await;
            if matches!(
                map.get(&key).map(|e| &e.status),
                Some(ImportStatus::Queued | ImportStatus::Importing)
            ) {
                // Already in flight — don't double-queue.
                return Ok(());
            }
            map.insert(key.clone(), entry(&session, ImportStatus::Queued, None));
        }

        let service = Arc::clone(self);
        tokio::spawn(async move {
            service.run_import(session, force).await;
        });
        Ok(())
    }

    /// Every tracked session's import status, for `GET /api/sessions/import`.
    pub async fn statuses(&self) -> Vec<StatusEntry> {
        self.status.lock().await.values().cloned().collect()
    }

    /// Run one import to completion, updating the status map at each transition.
    /// Errors are recorded as `failed` rather than propagated (this runs
    /// detached, so there is no caller to return them to).
    async fn run_import(&self, session: DiscoveredSession, force: bool) {
        let key = session_key(&session);
        self.set(&key, entry(&session, ImportStatus::Importing, None))
            .await;

        match self.do_import(&session, force).await {
            Ok(summary) => {
                self.set(&key, entry(&session, ImportStatus::Done, Some(summary)))
                    .await;
            }
            Err(e) => {
                tracing::warn!("session import '{}' failed: {e}", session.id);
                self.set(
                    &key,
                    entry(&session, ImportStatus::Failed, Some(e.to_string())),
                )
                .await;
            }
        }
    }

    /// The import itself: load the transcript, run the import use case, and
    /// render a one-line outcome summary. The use case builds its own chat
    /// client from the resolved endpoint, so nothing LLM-shaped is needed here.
    async fn do_import(
        &self,
        session: &DiscoveredSession,
        force: bool,
    ) -> Result<String, DomainError> {
        let use_case = self.container.memory_import_use_case()?;
        let transcript = self.discovery.load_transcript(session).await?;
        let outcome = use_case.execute(&transcript, force).await?;
        Ok(match outcome {
            ImportOutcome::Imported { report, .. } => {
                let written = report.memories_written;
                let mut summary = format!(
                    "{} memory{} written",
                    written,
                    if written == 1 { "" } else { "s" }
                );
                if report.conflicts_recorded > 0 {
                    summary.push_str(&format!(
                        ", {} conflict(s) recorded",
                        report.conflicts_recorded
                    ));
                }
                summary
            }
            ImportOutcome::AlreadyImported { .. } => "already imported".to_string(),
        })
    }

    /// Re-discover and resolve one session by its `(source, id)` identity. A
    /// missing session is a `NotFound` (→ 404 at the API), not an internal error.
    async fn find(&self, source: &str, id: &str) -> Result<DiscoveredSession, DomainError> {
        let sessions = self.discovery.discover().await?;
        sessions
            .into_iter()
            .find(|s| s.source.as_str() == source && s.id == id)
            .ok_or_else(|| {
                DomainError::not_found(format!(
                    "no discoverable session '{id}' from source '{source}'"
                ))
            })
    }

    async fn set(&self, key: &SessionKey, value: StatusEntry) {
        self.status.lock().await.insert(key.clone(), value);
    }
}

/// Stable identity for a discovered session (mirrors the TUI's `session_key`).
fn session_key(s: &DiscoveredSession) -> SessionKey {
    (s.source.as_str().to_string(), s.id.clone())
}

/// Normalize an `ImportedSession::source` string back to the bare source tag
/// (`claude` / `opencode` / `zed`) that [`session_key`] uses.
///
/// The stored source is heterogeneous by source: OpenCode/Zed record a
/// `"<tag>:<…>"` prefix, while Claude records the transcript file path. We match
/// the known tags (as a `"<tag>:"` prefix or an exact tag) and otherwise fall
/// back to `"claude"`, since a path is only ever a Claude transcript. Returning
/// the raw string when unknown would silently never match, dropping the ✓.
fn normalize_source_tag(stored: &str) -> String {
    for tag in [
        SessionSource::Claude.as_str(),
        SessionSource::OpenCode.as_str(),
        SessionSource::Zed.as_str(),
    ] {
        if stored == tag || stored.starts_with(&format!("{tag}:")) {
            return tag.to_string();
        }
    }
    // A bare path (no recognized tag prefix) is a Claude transcript.
    SessionSource::Claude.as_str().to_string()
}

/// Build a status-map entry for a session.
fn entry(s: &DiscoveredSession, status: ImportStatus, detail: Option<String>) -> StatusEntry {
    StatusEntry {
        source: s.source.as_str().to_string(),
        id: s.id.clone(),
        status,
        detail,
    }
}

/// Serialize a [`DiscoveredSession`] into the JSON DTO the API returns.
///
/// `DiscoveredSession` is not `Serialize` (its [`SessionLocator`] is an opaque
/// on-disk address we deliberately never expose to clients — imports are
/// requested by `(source, id)` and re-resolved server-side). This projects only
/// the display fields a picker shows.
pub fn session_to_json(s: &DiscoveredSession) -> serde_json::Value {
    // Kept in sync with the TUI list's columns: source, updated_at,
    // approx_tokens, title, plus cwd/message_count/preview for detail.
    let locator = match &s.locator {
        SessionLocator::File(_) => "file",
        SessionLocator::Sqlite { .. } => "sqlite",
    };
    serde_json::json!({
        "source": s.source.as_str(),
        "id": s.id,
        "title": s.display_title(),
        "cwd": s.cwd,
        "updated_at": s.updated_at,
        "message_count": s.message_count,
        "approx_tokens": s.approx_tokens,
        "tail_preview": s.tail_preview,
        "locator_kind": locator,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(source: SessionSource, id: &str) -> DiscoveredSession {
        DiscoveredSession {
            source,
            id: id.to_string(),
            title: format!("session {id}"),
            cwd: Some("/tmp/project".to_string()),
            updated_at: 1_700_000_000,
            message_count: 12,
            tail_preview: "…and that fixed it".to_string(),
            approx_tokens: 4_200,
            locator: SessionLocator::File(format!("/tmp/{id}.jsonl")),
        }
    }

    #[test]
    fn session_key_is_source_and_id() {
        let s = session(SessionSource::Zed, "abc");
        assert_eq!(session_key(&s), ("zed".to_string(), "abc".to_string()));
    }

    #[test]
    fn normalizes_prefixed_source_tags() {
        assert_eq!(normalize_source_tag("zed:/path/to.db"), "zed");
        assert_eq!(normalize_source_tag("opencode:/path/to.db"), "opencode");
        assert_eq!(normalize_source_tag("claude"), "claude");
    }

    #[test]
    fn normalizes_bare_path_to_claude() {
        // Claude records the transcript path, not a tag; it must still match.
        assert_eq!(
            normalize_source_tag("/Users/me/.claude/projects/p/abc.jsonl"),
            "claude"
        );
    }

    #[test]
    fn json_dto_omits_the_locator_path() {
        let value = session_to_json(&session(SessionSource::Claude, "abc"));
        assert_eq!(value["source"], "claude");
        assert_eq!(value["id"], "abc");
        assert_eq!(value["approx_tokens"], 4_200);
        assert_eq!(value["locator_kind"], "file");
        // The on-disk address must never reach a client.
        assert!(value.get("locator").is_none());
        assert!(!value.to_string().contains("/tmp/abc.jsonl"));
    }
}
