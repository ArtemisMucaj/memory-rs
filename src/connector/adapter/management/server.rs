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

use super::handlers;
use crate::connector::api::Container;
use crate::domain::DomainError;

/// Shared state handed to every management route handler.
#[derive(Clone)]
pub struct AppState {
    /// The dependency-injection container wiring adapters to use cases.
    pub container: Arc<Container>,
}

impl AppState {
    pub fn new(container: Arc<Container>) -> Self {
        Self { container }
    }
}

/// Assemble the management API [`Router`].
pub fn routes(state: AppState) -> Router {
    Router::new()
        .route("/", get(handlers::index))
        .route("/api", get(handlers::index))
        .route("/health", get(handlers::health))
        .route("/api/search", get(handlers::search))
        .route("/api/memory", get(handlers::list_items))
        .route(
            "/api/memory/{id}",
            get(handlers::show).delete(handlers::delete),
        )
        .route("/api/tree", get(handlers::tree))
        .route("/api/sessions", get(handlers::sessions))
        .route("/api/stats", get(handlers::stats))
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
        .route("/api/import", post(handlers::import))
        .route("/api/resources", post(handlers::add_resource))
        .route("/api/dream", post(handlers::dream))
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
    let app = routes(AppState::new(container)).nest_service("/mcp", mcp);

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
        routes(AppState::new(Arc::new(container)))
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

    #[tokio::test]
    async fn stats_endpoint_returns_counts() {
        let dir = tempfile::tempdir().unwrap();
        let app = test_app(dir.path());
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["total_items"], 0);
        assert!(json["items_by_kind"].is_array());
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
