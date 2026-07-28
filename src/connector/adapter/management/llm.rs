//! LLM endpoint configuration + model discovery.
//!
//! - `GET    /api/llm/endpoints`        — registered endpoints and which are active
//! - `PUT    /api/llm/endpoints/{name}` — register or update one endpoint
//! - `DELETE /api/llm/endpoints/{name}` — remove one endpoint
//! - `POST   /api/llm/active`           — pick the active endpoint for a role
//! - `GET    /api/llm/models?endpoint=` — discover models from a server
//!
//! Configuration is per-service on purpose: memory and code intelligence may
//! want different backends (a small local model for one, a hosted model for the
//! other), so each owns its own `config.json` rather than sharing one via
//! environment variables. The `OPENAI_*` environment variables remain the
//! fallback when nothing is registered here.
//!
//! **Chat and embeddings resolve independently.** `active` is the shared
//! default; `active_chat` / `active_embedding` override it per role, so a
//! remote chat model can pair with local embeddings. Note that the embedding
//! *dimension* is pinned to the database on first open — switching to an
//! embedding model of a different dimension is rejected at open time, so the
//! response flags the pinned model to let a UI warn before the damage is done.

use std::time::Duration;

use axum::extract::{Path, Query, State};
use axum::Json;
use openai_rs::{Endpoint, ModelCatalog, OpenAiModelCatalog};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::error::{ApiError, ApiResult};
use super::server::AppState;
use crate::connector::adapter::{MemoryConfig, OpenAiConfig, OpenAiEndpoint, COPILOT_ENDPOINT};
use crate::domain::DomainError;

/// Which resolution slot an endpoint is being bound to.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActiveRole {
    /// The shared default used by whichever role has no override.
    #[default]
    Shared,
    /// Extraction / summarization / dreaming.
    Chat,
    /// Semantic recall.
    Embedding,
}

/// Load the config document, or default when the file is absent.
fn load(state: &AppState) -> Result<MemoryConfig, DomainError> {
    MemoryConfig::load(state.container.data_dir())
}

/// Persist a mutated config document off the async runtime — `save` does
/// blocking filesystem I/O, so running it inline would stall the request thread.
async fn save(state: &AppState, config: MemoryConfig) -> Result<(), DomainError> {
    let data_dir = state.container.data_dir().to_string();
    tokio::task::spawn_blocking(move || config.save(&data_dir))
        .await
        .map_err(|e| DomainError::internal(format!("config write task panicked: {e}")))?
}

/// Render the endpoints document. API keys are reported as a boolean, never
/// echoed: this API is unauthenticated on loopback and a UI only needs to know
/// whether a key is set, not what it is.
fn endpoints_json(config: &MemoryConfig) -> Value {
    let openai = config.openai.clone().unwrap_or_default();
    let endpoints: Vec<Value> = openai
        .endpoints
        .iter()
        .map(|(name, e)| {
            json!({
                "name": name,
                "base_url": e.base_url,
                "model": e.model,
                "embedding_model": e.embedding_model,
                "has_api_key": e.api_key.as_ref().is_some_and(|k| !k.is_empty()),
            })
        })
        .collect();

    json!({
        "endpoints": endpoints,
        "active": openai.active,
        "active_chat": openai.active_chat,
        "active_embedding": openai.active_embedding,
        // The dimension the database is pinned to. Switching to an embedding
        // model that emits a different width is rejected on the next open, so a
        // UI should warn rather than let the user strand their store.
        "pinned_embedding": config.embedding.as_ref().map(|e| json!({
            "model": e.model,
            "dimensions": e.dimensions,
        })),
        // Copilot is bindable by its reserved name rather than being a
        // registered endpoint, so it is reported separately.
        "copilot": {
            "endpoint_name": COPILOT_ENDPOINT,
            "authenticated": config.copilot.as_ref().is_some_and(|c| c.is_authenticated()),
            "model": config.copilot.as_ref().and_then(|c| c.model.clone()),
        },
    })
}

/// `GET /api/llm/endpoints` — registered endpoints plus the active selections.
pub async fn list_endpoints(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    Ok(Json(endpoints_json(&load(&state)?)))
}

/// Body for `PUT /api/llm/endpoints/{name}`.
#[derive(Debug, Deserialize)]
pub struct UpsertEndpointBody {
    /// Base URL with no `/v1` suffix — the clients append the version and route
    /// themselves, so a trailing `/v1` here produces `/v1/v1/...`. Omitted for
    /// the reserved `copilot` name, whose URL is fixed by the provider.
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub embedding_model: Option<String>,
    /// Omit to leave an existing key untouched; send `""` to clear it.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Bind this endpoint to a role in the same call.
    #[serde(default)]
    pub set_active: Option<ActiveRole>,
}

/// Strip a trailing slash and `/v1` so a user- or probe-supplied `.../v1`
/// doesn't become `/v1/v1/models`.
fn normalize_base(raw: &str) -> String {
    let mut s = raw.trim().to_string();
    while s.ends_with('/') {
        s.pop();
    }
    if s.ends_with("/v1") {
        s.truncate(s.len() - 3);
    }
    while s.ends_with('/') {
        s.pop();
    }
    s
}

/// `PUT /api/llm/endpoints/{name}` — register or update one endpoint.
pub async fn upsert_endpoint(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(body): Json<UpsertEndpointBody>,
) -> ApiResult<Json<Value>> {
    if name.trim().is_empty() {
        return Err(ApiError::from(DomainError::invalid_input(
            "endpoint name must not be empty",
        )));
    }

    // `copilot` is reserved: it has no user-supplied base URL or key, so an
    // upsert against that name only pins the model (and optionally binds a
    // role) rather than registering an OpenAI endpoint.
    if name == COPILOT_ENDPOINT {
        let mut config = load(&state)?;
        let mut copilot = config.copilot.take().unwrap_or_default();
        if let Some(model) = body.model {
            copilot.model = Some(model);
        }
        config.copilot = Some(copilot);
        if let Some(role) = body.set_active {
            let mut openai = config.openai.take().unwrap_or_default();
            bind_active(&mut openai, role, Some(COPILOT_ENDPOINT.to_string()));
            config.openai = Some(openai);
        }
        save(&state, config.clone()).await?;
        return Ok(Json(endpoints_json(&config)));
    }

    let base_url = normalize_base(body.base_url.as_deref().unwrap_or_default());
    if base_url.is_empty() {
        return Err(ApiError::from(DomainError::invalid_input(
            "base_url must not be empty",
        )));
    }

    let mut config = load(&state)?;
    let mut openai = config.openai.take().unwrap_or_default();

    let existing = openai.endpoints.get(&name).cloned().unwrap_or_default();
    let api_key = match body.api_key {
        // An explicit empty string clears the key; omitting the field keeps it.
        Some(k) if k.is_empty() => None,
        Some(k) => Some(k),
        None => existing.api_key,
    };

    openai.endpoints.insert(
        name.clone(),
        OpenAiEndpoint {
            base_url,
            model: body.model.or(existing.model),
            embedding_model: body.embedding_model.or(existing.embedding_model),
            api_key,
        },
    );

    if let Some(role) = body.set_active {
        bind_active(&mut openai, role, Some(name.clone()));
    }

    config.openai = Some(openai);
    save(&state, config.clone()).await?;
    Ok(Json(endpoints_json(&config)))
}

/// `DELETE /api/llm/endpoints/{name}` — remove an endpoint.
///
/// Any role pointing at it is cleared too, so the config never names a missing
/// endpoint (which would silently fall through to the environment).
pub async fn delete_endpoint(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<Json<Value>> {
    let mut config = load(&state)?;
    let mut openai = config.openai.take().unwrap_or_default();

    if openai.endpoints.remove(&name).is_none() {
        config.openai = Some(openai);
        return Err(ApiError::from(DomainError::not_found(format!(
            "no LLM endpoint named '{name}'"
        ))));
    }
    for slot in [
        &mut openai.active,
        &mut openai.active_chat,
        &mut openai.active_embedding,
    ] {
        if slot.as_deref() == Some(name.as_str()) {
            *slot = None;
        }
    }

    config.openai = Some(openai);
    save(&state, config.clone()).await?;
    Ok(Json(endpoints_json(&config)))
}

/// Body for `POST /api/llm/active`.
#[derive(Debug, Deserialize)]
pub struct SetActiveBody {
    /// Endpoint name, or `null` to clear the role back to the shared default
    /// (and, for `shared`, back to the `OPENAI_*` environment).
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub role: ActiveRole,
}

fn bind_active(openai: &mut OpenAiConfig, role: ActiveRole, name: Option<String>) {
    match role {
        ActiveRole::Shared => openai.active = name,
        ActiveRole::Chat => openai.active_chat = name,
        ActiveRole::Embedding => openai.active_embedding = name,
    }
}

/// `POST /api/llm/active` — bind (or clear) the endpoint used for a role.
pub async fn set_active(
    State(state): State<AppState>,
    Json(body): Json<SetActiveBody>,
) -> ApiResult<Json<Value>> {
    let mut config = load(&state)?;
    let mut openai = config.openai.take().unwrap_or_default();

    // Refuse to point a role at an endpoint that isn't registered: the resolver
    // treats a dangling name as "unset" and silently falls back, which reads as
    // the setting having been ignored.
    if let Some(name) = body.name.as_deref() {
        // `copilot` is reserved and never appears in `endpoints`, but it IS a
        // valid binding — the container resolves it to the Copilot backend.
        if name != COPILOT_ENDPOINT && !openai.endpoints.contains_key(name) {
            config.openai = Some(openai);
            return Err(ApiError::from(DomainError::not_found(format!(
                "no LLM endpoint named '{name}'"
            ))));
        }
    }

    bind_active(&mut openai, body.role, body.name);
    config.openai = Some(openai);
    save(&state, config.clone()).await?;
    Ok(Json(endpoints_json(&config)))
}

// ── GitHub Copilot ───────────────────────────────────────────────────────────

/// `POST /api/llm/copilot/login` — begin the OAuth device flow.
///
/// Returns immediately with the `user_code` + `verification_uri` to show the
/// user; the server polls GitHub in the background and persists the token on
/// success.
pub async fn copilot_login_start(State(state): State<AppState>) -> Json<Value> {
    Json(json!(state.copilot_login.start().await))
}

/// `GET /api/llm/copilot/login` — the current login status
/// (`idle` / `pending` / `authorized` / `failed`).
pub async fn copilot_login_status(State(state): State<AppState>) -> Json<Value> {
    Json(json!(state.copilot_login.status().await))
}

#[derive(Debug, Deserialize)]
pub struct ModelsParams {
    /// Registered endpoint to query. Defaults to the active chat endpoint, then
    /// the shared default, then the `OPENAI_*` environment. Pass the reserved
    /// name `copilot` to enumerate the Copilot subscription's models.
    #[serde(default)]
    pub endpoint: Option<String>,
    /// Query this base URL directly instead of a registered endpoint — lets a
    /// UI validate a server before saving it.
    #[serde(default)]
    pub base_url: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vendor: Option<String>,
}

/// `GET /api/llm/models` — enumerate the models a server offers.
///
/// Used to populate a model picker, so it resolves an endpoint the same way the
/// chat client does and reports a failure as a 400 (bad address / unreachable
/// server) rather than an opaque 500.
pub async fn models(
    State(state): State<AppState>,
    Query(params): Query<ModelsParams>,
) -> ApiResult<Json<Value>> {
    let config = load(&state)?;
    let openai = config.openai.clone().unwrap_or_default();

    // Copilot has its own catalog with richer metadata, and no base URL to
    // query — it's addressed by the reserved endpoint name.
    if params.endpoint.as_deref() == Some(COPILOT_ENDPOINT) {
        let models = crate::connector::adapter::copilot::list_models(
            &config.copilot.clone().unwrap_or_default(),
        )
        .await?;
        let list: Vec<Value> = models
            .into_iter()
            .map(|m| {
                json!({
                    "id": m.id,
                    "vendor": m.vendor,
                    "name": m.name,
                })
            })
            .collect();
        return Ok(Json(json!({ "base_url": COPILOT_ENDPOINT, "models": list })));
    }

    let (base_url, api_key) = if let Some(raw) = params.base_url.as_deref() {
        (normalize_base(raw), None)
    } else {
        let name = params
            .endpoint
            .clone()
            .or_else(|| openai.active_chat.clone())
            .or_else(|| openai.active.clone());
        match name.and_then(|n| openai.endpoints.get(&n).cloned()) {
            Some(e) => (e.base_url, e.api_key),
            // Nothing registered — fall back to the environment, matching how
            // the chat client resolves when no endpoint is configured.
            None => (
                normalize_base(
                    &std::env::var("OPENAI_BASE_URL")
                        .unwrap_or_else(|_| "http://localhost:1234".to_string()),
                ),
                std::env::var("OPENAI_API_KEY").ok().filter(|k| !k.is_empty()),
            ),
        }
    };

    let endpoint = Endpoint::new(base_url.clone()).with_optional_api_key(api_key);
    let catalog = OpenAiModelCatalog::new(&endpoint)
        .map_err(|e| ApiError::from(DomainError::invalid_input(format!("{base_url}: {e}"))))?
        .with_timeout(Duration::from_secs(15));

    let models = catalog.list_models().await.map_err(|e| {
        ApiError::from(DomainError::invalid_input(format!(
            "could not list models from {base_url}: {e}"
        )))
    })?;

    let list: Vec<ModelInfo> = models
        .into_iter()
        .map(|m| ModelInfo {
            id: m.id.clone(),
            vendor: m.vendor.clone(),
        })
        .collect();

    Ok(Json(json!({ "base_url": base_url, "models": list })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_v1_and_slashes() {
        assert_eq!(normalize_base("http://localhost:1234/v1"), "http://localhost:1234");
        assert_eq!(normalize_base("http://localhost:1234/v1/"), "http://localhost:1234");
        assert_eq!(normalize_base("http://localhost:1234/"), "http://localhost:1234");
        assert_eq!(normalize_base("  http://localhost:1234  "), "http://localhost:1234");
    }

    #[test]
    fn normalize_leaves_a_bare_base_alone() {
        assert_eq!(normalize_base("https://api.openai.com"), "https://api.openai.com");
    }

    #[test]
    fn bind_active_targets_the_named_role() {
        let mut openai = OpenAiConfig::default();

        bind_active(&mut openai, ActiveRole::Chat, Some("remote".into()));
        assert_eq!(openai.active_chat.as_deref(), Some("remote"));
        assert!(openai.active.is_none());
        assert!(openai.active_embedding.is_none());

        bind_active(&mut openai, ActiveRole::Embedding, Some("local".into()));
        assert_eq!(openai.active_embedding.as_deref(), Some("local"));

        // Clearing a role leaves the others intact.
        bind_active(&mut openai, ActiveRole::Chat, None);
        assert!(openai.active_chat.is_none());
        assert_eq!(openai.active_embedding.as_deref(), Some("local"));
    }

    #[test]
    fn endpoints_json_never_echoes_the_api_key() {
        let mut openai = OpenAiConfig::default();
        openai.endpoints.insert(
            "hosted".into(),
            OpenAiEndpoint {
                base_url: "https://api.openai.com".into(),
                model: Some("gpt-4".into()),
                embedding_model: None,
                api_key: Some("sk-secret-value".into()),
            },
        );
        let config = MemoryConfig {
            openai: Some(openai),
            ..Default::default()
        };

        let value = endpoints_json(&config);
        let text = value.to_string();
        assert!(!text.contains("sk-secret-value"));
        assert_eq!(value["endpoints"][0]["has_api_key"], true);
    }
}
