//! Management HTTP API server (axum).
//!
//! Builds an [`axum::Router`] serving a REST/JSON API over the shared
//! controllers, and runs it until ctrl-c. A native app (or any HTTP client)
//! drives every memory operation through these endpoints; the MCP server is
//! mounted on the same process so one `serve` gives both surfaces.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::routing::{delete, get, post};
use axum::Router;
use tower_http::cors::CorsLayer;

use super::copilot_login::CopilotLoginService;
use super::dream::DreamService;
use super::session_import::SessionImportService;
use super::{dream_routes, handlers, llm, sessions};
use crate::connector::api::Container;
use crate::domain::DomainError;

/// Shared state handed to every management route handler.
#[derive(Clone)]
pub struct AppState {
    /// The dependency-injection container wiring adapters to use cases.
    pub container: Arc<Container>,
    /// Session discovery + background imports. Shared (not per-request) so an
    /// import's status survives the request that queued it.
    pub sessions: Arc<SessionImportService>,
    /// The dream scheduler. Shared so the background loop and the status/trigger
    /// endpoints observe the same `running` flag and live config.
    pub dream: Arc<DreamService>,
    /// GitHub Copilot device-flow login. Shared so the background poll that
    /// persists the token and the status endpoint see one session.
    pub copilot_login: Arc<CopilotLoginService>,
}

impl AppState {
    /// Build the shared state.
    ///
    /// Fallible because the dream scheduler needs the resolved chat endpoint and
    /// the memory store up front; a server that cannot schedule dreams should
    /// fail at start-up rather than 500 on the first status poll.
    pub fn new(container: Arc<Container>) -> Result<Self, DomainError> {
        let sessions = SessionImportService::build(Arc::clone(&container));
        let dream = DreamService::build(&container)?;
        let copilot_login = CopilotLoginService::new(container.data_dir().to_string());
        Ok(Self {
            container,
            sessions,
            dream,
            copilot_login,
        })
    }
}

/// Assemble the management API [`Router`].
pub fn routes(state: AppState) -> Router {
    Router::new()
        .route("/", get(handlers::index))
        .route("/api", get(handlers::index))
        .route("/health", get(handlers::health))
        .route("/api/search", get(handlers::search))
        .route("/api/memory", get(handlers::list_memories))
        .route("/api/memory/{id}", get(handlers::show).delete(handlers::delete))
        .route("/api/entities", get(handlers::entities))
        .route("/api/entities/{id}", get(handlers::entity))
        .route("/api/tree", get(handlers::tree))
        .route("/api/sessions", get(handlers::sessions))
        .route("/api/resume", get(handlers::resume))
        .route(
            "/api/namespaces",
            get(handlers::list_namespaces).post(handlers::create_namespace),
        )
        .route(
            "/api/namespaces/{name}",
            get(handlers::show_namespace).delete(handlers::delete_namespace),
        )
        .route(
            "/api/namespaces/{name}/projects",
            post(handlers::assign_project),
        )
        .route(
            "/api/namespaces/{name}/projects/{project}",
            delete(handlers::unassign_project),
        )
        // Session discovery + background import (what the TUI's Import screen
        // does in-process). Discovery is under `/discover` because `/api/sessions`
        // above already serves the *imported* sessions — a different set.
        .route("/api/sessions/discover", get(sessions::discover))
        .route("/api/sessions/transcript", get(sessions::transcript))
        .route(
            "/api/sessions/import",
            get(sessions::import_status).post(sessions::import),
        )
        .route("/api/import", post(handlers::import))
        .route("/api/resources", post(handlers::add_resource))
        // Dream scheduler: status, background trigger, and live settings.
        // `POST` starts a cycle and returns 202 rather than blocking for the
        // many minutes a full consolidation can take.
        .route(
            "/api/dream",
            get(dream_routes::status).post(dream_routes::trigger),
        )
        .route(
            "/api/dream/config",
            axum::routing::put(dream_routes::update_config),
        )
        // LLM endpoint configuration + model discovery. Per-service on purpose:
        // memory and code intelligence may want different backends.
        .route("/api/llm/endpoints", get(llm::list_endpoints))
        .route(
            "/api/llm/endpoints/{name}",
            axum::routing::put(llm::upsert_endpoint).delete(llm::delete_endpoint),
        )
        .route("/api/llm/active", post(llm::set_active))
        .route("/api/llm/models", get(llm::models))
        // Per-usage model selection: each LLM job this server runs can name its
        // own endpoint + model, falling back to the shared role.
        .route("/api/llm/usages", get(llm::list_usages))
        .route("/api/llm/usages/{id}", axum::routing::put(llm::set_usage))
        // GitHub Copilot device-flow login, so a GUI can authenticate without a
        // terminal command (the bundled binary isn't on the user's PATH).
        .route(
            "/api/llm/copilot/login",
            get(llm::copilot_login_status).post(llm::copilot_login_start),
        )
        // Allow a local native app (different origin) to call the API.
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Bind and serve the management API (plus MCP over HTTP at `/mcp`) until
/// ctrl-c.
///
/// `public` binds `0.0.0.0` (reachable off-host) instead of loopback — off by
/// default, since the API is unauthenticated.
pub async fn serve(container: Container, port: u16, public: bool) -> Result<(), DomainError> {
    let container = Arc::new(container);

    // Mount the MCP streamable-HTTP service at /mcp on the same process, so one
    // `serve` gives a native app both the REST API and MCP over HTTP.
    let mcp = super::super::mcp::http_service(Arc::clone(&container));
    let state = AppState::new(container)?;

    // Run the dream scheduler alongside the server: it harvests finished
    // sessions on a sweep and consolidates when a full cycle is due. Detached,
    // so it lives as long as the process; the shared `DreamService` is what the
    // status/trigger endpoints observe.
    tokio::spawn(Arc::clone(&state.dream).run_scheduler());

    let app = routes(state).nest_service("/mcp", mcp);

    let bind_addr: [u8; 4] = if public { [0, 0, 0, 0] } else { [127, 0, 0, 1] };
    let addr = SocketAddr::from((bind_addr, port));
    let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
        DomainError::internal(format!("failed to bind management API to {addr}: {e}"))
    })?;

    tracing::info!("memory-rs management API listening on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| DomainError::internal(format!("management API server error: {e}")))
}

/// Resolve when the process receives ctrl-c, for graceful shutdown.
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutting down management API");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connector::api::ContainerConfig;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt; // for `oneshot`

    fn test_app(dir: &std::path::Path) -> Router {
        let container = Container::new(ContainerConfig {
            data_dir: dir.to_str().unwrap().to_string(),
            embedding_dimensions: 4,
            openai_endpoint: None,
        })
        .unwrap();
        routes(AppState::new(Arc::new(container)).unwrap())
    }

    async fn body_json(resp: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn health_reports_ok() {
        let dir = tempfile::tempdir().unwrap();
        let app = test_app(dir.path());
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["status"], "ok");
    }

    /// Contract test. The list envelope's key is `memories`. The native app
    /// decodes `/api/memory` into a struct whose keys are required; if the
    /// key is renamed server-side the app renders an empty dashboard rather
    /// than failing, so the rename has to fail *here*.
    #[tokio::test]
    async fn memory_endpoint_returns_a_memories_envelope() {
        let dir = tempfile::tempdir().unwrap();
        let app = test_app(dir.path());
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/memory")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert!(
            json.get("memories").is_some_and(|c| c.is_array()),
            "GET /api/memory must carry a `memories` array, got: {json}",
        );
    }

    #[tokio::test]
    async fn namespace_create_and_list_over_http() {
        let dir = tempfile::tempdir().unwrap();
        let app = test_app(dir.path());

        // Create.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/namespaces")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"name":"payments"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_json(resp).await["created"], true);

        // List shows it.
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/namespaces")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let json = body_json(resp).await;
        assert_eq!(json["namespaces"][0]["name"], "payments");
    }
}
