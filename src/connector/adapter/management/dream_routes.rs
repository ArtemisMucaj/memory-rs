//! Dream scheduler status, manual trigger, and live settings.
//!
//! - `GET  /api/dream`        — scheduler config + whether a sweep is running
//! - `POST /api/dream`        — start one harvest sweep in the background (202)
//! - `PUT  /api/dream/config` — partial update, applied live and persisted
//!
//! `POST /api/dream` returns as soon as the sweep is queued rather than
//! holding the connection open. The CLI's `memory-rs dream` still runs one
//! synchronously in the foreground.

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde_json::{json, Value};

use super::dream::DreamConfigPatch;
use super::error::ApiResult;
use super::server::AppState;

/// Render the scheduler's config + liveness as the status payload.
fn status_json(state: &AppState, running: bool) -> Value {
    let cfg = state.dream.config();
    json!({
        "enabled": cfg.dream_enabled(),
        "interval_hours": cfg.dream_interval_hours(),
        "session_idle_minutes": cfg.session_idle_minutes(),
        "auto_import": cfg.auto_import(),
        "running": running,
    })
}

/// `GET /api/dream` — the scheduler's settings and whether a sweep is in
/// flight.
pub async fn status(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let running = state.dream.is_running();
    Ok(Json(status_json(&state, running)))
}

/// `POST /api/dream` — start one harvest sweep in the background.
///
/// `202 Accepted` with `{"started": true}` when a sweep was queued, or
/// `{"started": false}` when one is already running.
pub async fn trigger(State(state): State<AppState>) -> ApiResult<(StatusCode, Json<Value>)> {
    let started = state.dream.trigger();
    let mut body = status_json(&state, true);
    if let Some(obj) = body.as_object_mut() {
        obj.insert("started".to_string(), json!(started));
    }
    Ok((StatusCode::ACCEPTED, Json(body)))
}

/// `PUT /api/dream/config` — partial update of the scheduler's settings.
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
