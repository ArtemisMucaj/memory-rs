//! Dream scheduler status, manual trigger, and live settings.
//!
//! - `GET  /api/dream`        — scheduler config + whether a cycle is running
//! - `POST /api/dream`        — start one cycle in the **background** (202)
//! - `PUT  /api/dream/config` — partial update, applied live and persisted
//!
//! `POST /api/dream` returns as soon as the cycle is queued rather than holding
//! the connection open: a full consolidation over a large store runs for many
//! minutes of LLM calls. Poll `GET /api/dream` for `running`. The CLI's
//! `memory-rs dream` still runs one synchronously in the foreground.

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde_json::{json, Value};

use super::dream::DreamConfigPatch;
use super::error::ApiResult;
use super::server::AppState;

/// Render the scheduler's config + liveness as the status payload.
fn status_json(
    state: &AppState,
    running: bool,
    last_run: Option<crate::domain::DreamRun>,
) -> Value {
    let cfg = state.dream.config();
    json!({
        "enabled": cfg.dream_enabled(),
        "interval_hours": cfg.dream_interval_hours(),
        "session_idle_minutes": cfg.session_idle_minutes(),
        "auto_import": cfg.auto_import(),
        "running": running,
        "last_run": last_run.map(|r| json!({
            "started_at": r.started_at,
            "finished_at": r.finished_at,
            "status": r.status,
            "sessions_imported": r.sessions_imported,
            "operations_applied": r.operations_applied,
        })),
    })
}

/// `GET /api/dream` — the scheduler's settings, whether a cycle is in flight,
/// and the last recorded run.
pub async fn status(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let running = state.dream.is_running();
    let last_run = state.dream.last_run().await;
    Ok(Json(status_json(&state, running, last_run)))
}

/// `POST /api/dream` — start one dream cycle in the background.
///
/// `202 Accepted` with `{"started": true}` when a cycle was queued, or
/// `{"started": false}` when one is already running (not an error: the caller's
/// intent — "a cycle is running" — already holds).
pub async fn trigger(State(state): State<AppState>) -> ApiResult<(StatusCode, Json<Value>)> {
    let started = state.dream.trigger();
    let last_run = state.dream.last_run().await;
    let mut body = status_json(&state, true, last_run);
    if let Some(obj) = body.as_object_mut() {
        obj.insert("started".to_string(), json!(started));
    }
    Ok((StatusCode::ACCEPTED, Json(body)))
}

/// `PUT /api/dream/config` — partial update of the scheduler's settings.
///
/// Applied to the running scheduler immediately (it reads a fresh snapshot each
/// tick) and persisted to `config.json`. Returns the merged effective config.
pub async fn update_config(
    State(state): State<AppState>,
    Json(patch): Json<DreamConfigPatch>,
) -> ApiResult<Json<Value>> {
    let merged = state.dream.update_config(patch).await?;
    Ok(Json(json!({
        "dream_enabled": merged.dream_enabled(),
        "dream_interval_hours": merged.dream_interval_hours(),
        "session_idle_minutes": merged.session_idle_minutes(),
        "auto_import": merged.auto_import(),
    })))
}
