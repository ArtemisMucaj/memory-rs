//! Session discovery + background import endpoints.
//!
//! These expose, over REST, what the TUI's Import screen does in-process:
//! discover finished assistant sessions (Claude Code / OpenCode / Zed), preview
//! a transcript, and import a chosen session in the background so a native
//! client can drive the same flow without holding a connection open for a whole
//! LLM extraction.
//!
//! - `GET  /api/sessions/discover`   — discover importable sessions (newest first)
//! - `GET  /api/sessions/transcript` — one session's full transcript (`?source=&id=`)
//! - `POST /api/sessions/import`     — queue a background import (`{source,id,force?}`)
//! - `GET  /api/sessions/import`     — per-session import status map
//!
//! Discovery lives under `/api/sessions/discover` rather than `/api/sessions`
//! because this server already serves *imported* sessions at `/api/sessions` —
//! a different set (store records, not on-disk transcripts).

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use super::error::ApiResult;
use super::server::AppState;
use super::session_import::session_to_json;

/// `GET /api/sessions/discover` — discover finished sessions across every
/// source, newest first. Each entry carries the display fields a picker shows;
/// the separate status map (`GET /api/sessions/import`) says which are imported.
pub async fn discover(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let sessions = state.sessions.discover().await?;
    let list: Vec<Value> = sessions.iter().map(session_to_json).collect();
    Ok(Json(json!({ "count": list.len(), "sessions": list })))
}

/// Query params identifying one discovered session by its stable identity.
#[derive(Debug, Deserialize)]
pub struct SessionRef {
    /// Discovery source: `claude`, `opencode`, or `zed`.
    pub source: String,
    /// The session's stable id (as returned by `GET /api/sessions/discover`).
    pub id: String,
}

/// `GET /api/sessions/transcript?source=&id=` — the full, per-turn transcript
/// of one discovered session, for a preview pane before importing.
pub async fn transcript(
    State(state): State<AppState>,
    Query(params): Query<SessionRef>,
) -> ApiResult<Json<Value>> {
    let transcript = state
        .sessions
        .transcript(&params.source, &params.id)
        .await?;
    Ok(Json(json!({
        "id": transcript.id,
        "source": transcript.source,
        "project": transcript.project,
        "message_count": transcript.messages.len(),
        "messages": transcript.messages,
    })))
}

/// Body for `POST /api/sessions/import`.
#[derive(Debug, Deserialize)]
pub struct ImportRequest {
    pub source: String,
    pub id: String,
    /// Re-import even if the session is already in the store.
    #[serde(default)]
    pub force: bool,
}

/// `POST /api/sessions/import` — queue a background import of one session.
///
/// Returns `202 Accepted` immediately; the import runs on a detached task, so
/// it continues after this request completes and the client can poll
/// `GET /api/sessions/import` for progress.
pub async fn import(
    State(state): State<AppState>,
    Json(req): Json<ImportRequest>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    state
        .sessions
        .import(&req.source, &req.id, req.force)
        .await?;
    Ok((StatusCode::ACCEPTED, Json(json!({ "queued": true }))))
}

/// `GET /api/sessions/import` — the import status of every tracked session,
/// keyed by `(source, id)`. Mirrors the TUI's per-row markers so a client can
/// render `queued`/`importing`/`done`/`failed`/`already_imported`.
pub async fn import_status(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let statuses = state.sessions.statuses().await;
    Ok(Json(
        json!({ "count": statuses.len(), "statuses": statuses }),
    ))
}
