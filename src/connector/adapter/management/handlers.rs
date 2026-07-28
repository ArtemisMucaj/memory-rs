//! Management API request handlers.
//!
//! Each handler is a thin adapter: extract the shared [`AppState`], parse
//! query/body params, call a [`controller`](crate::connector::api::controller)
//! function, and serialize the returned domain data to JSON. All operation
//! logic lives in the controllers, shared with the CLI and MCP surfaces.

use axum::extract::{Path, Query, State};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use super::error::{ApiError, ApiResult};
use super::server::AppState;
use crate::connector::api::controller::{
    self, DeleteOutcome, SearchOutcome, SearchScope, ShowOutcome,
};
use crate::domain::{MemoryItem, MemoryKind, MemoryNode};

/// `GET /health` — liveness + version.
pub async fn health() -> Json<Value> {
    Json(json!({ "status": "ok", "version": env!("CARGO_PKG_VERSION") }))
}

/// `GET /api` — a small index of the available endpoints.
pub async fn index() -> Json<Value> {
    Json(json!({
        "name": "memory-rs management API",
        "version": env!("CARGO_PKG_VERSION"),
        "endpoints": [
            "GET  /health",
            "GET  /api/search?q=&kind=&project=&namespace=&limit=",
            "GET  /api/memory?kind=",
            "GET  /api/memory/{id}",
            "DELETE /api/memory/{id}",
            "GET  /api/tree?uri=",
            "GET  /api/sessions",
            "GET  /api/sessions/discover",
            "GET  /api/sessions/transcript?source=&id=",
            "POST /api/sessions/import  {source, id, force?}",
            "GET  /api/sessions/import",
            "GET  /api/stats",
            "GET  /api/namespaces",
            "POST /api/namespaces  {name}",
            "DELETE /api/namespaces/{name}",
            "GET  /api/namespaces/{name}",
            "POST /api/namespaces/{name}/projects  {project}",
            "DELETE /api/namespaces/{name}/projects/{project}",
            "POST /api/import  {path, force?}",
            "POST /api/resources  {source, name?}",
            "GET  /api/dream",
            "POST /api/dream",
            "PUT  /api/dream/config  {dream_enabled?, dream_interval_hours?, session_idle_minutes?, auto_import?}",
            "GET  /api/llm/endpoints",
            "PUT  /api/llm/endpoints/{name}  {base_url, model?, embedding_model?, api_key?, set_active?}",
            "DELETE /api/llm/endpoints/{name}",
            "POST /api/llm/active  {name?, role?}",
            "GET  /api/llm/models?endpoint=&base_url="
        ]
    }))
}

#[derive(Debug, Deserialize)]
pub struct SearchParams {
    /// Query string (`q`).
    pub q: String,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

/// `GET /api/search` — hybrid search, scoped to all / a project / a namespace.
pub async fn search(
    State(state): State<AppState>,
    Query(params): Query<SearchParams>,
) -> ApiResult<Json<Value>> {
    let kind = parse_kind(params.kind.as_deref())?;
    let scope = match (params.namespace, params.project) {
        (Some(ns), _) => SearchScope::Namespace(ns),
        (None, Some(p)) => SearchScope::Project(p),
        (None, None) => SearchScope::All,
    };
    let limit = params.limit.unwrap_or(10).min(100);

    match controller::search(&state.container, &params.q, kind, &scope, limit).await? {
        SearchOutcome::Hits(hits) => Ok(Json(json!({
            "results": hits.iter().map(item_with_score).collect::<Vec<_>>(),
        }))),
        SearchOutcome::EmptyNamespace(ns) => Ok(Json(json!({
            "results": [],
            "note": format!("namespace '{ns}' has no member projects"),
        }))),
    }
}

#[derive(Debug, Deserialize)]
pub struct ListParams {
    #[serde(default)]
    pub kind: Option<String>,
}

/// `GET /api/memory` — list items, optionally filtered by kind.
pub async fn list_items(
    State(state): State<AppState>,
    Query(params): Query<ListParams>,
) -> ApiResult<Json<Value>> {
    let kind = parse_kind(params.kind.as_deref())?;
    let items = controller::list_items(&state.container, kind).await?;
    Ok(Json(json!({ "items": items })))
}

/// `GET /api/memory/{id}` — show an item (by id or `kind/name`) or a node URI.
pub async fn show(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    match controller::show(&state.container, &id).await? {
        ShowOutcome::Node(node) => Ok(Json(json!({ "type": "node", "node": node_json(&node) }))),
        ShowOutcome::Item(item) => Ok(Json(json!({ "type": "item", "item": item }))),
        ShowOutcome::Many(items) => Ok(Json(json!({ "type": "ambiguous", "matches": items }))),
        ShowOutcome::NotFound => Err(ApiError::from(crate::domain::DomainError::not_found(
            format!("no memory matches '{id}'"),
        ))),
    }
}

/// `DELETE /api/memory/{id}` — delete an item by id or unique `kind/name`.
pub async fn delete(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    match controller::delete(&state.container, &id).await? {
        DeleteOutcome::Deleted => Ok(Json(json!({ "deleted": true }))),
        DeleteOutcome::Ambiguous(items) => Ok(Json(json!({
            "deleted": false,
            "reason": "ambiguous kind/name across projects; delete by id",
            "matches": items,
        }))),
        DeleteOutcome::NotFound => Err(ApiError::from(crate::domain::DomainError::not_found(
            format!("no memory item '{id}'"),
        ))),
    }
}

#[derive(Debug, Deserialize)]
pub struct TreeParams {
    #[serde(default)]
    pub uri: Option<String>,
}

/// `GET /api/tree` — list a directory's children (root when `uri` is omitted).
pub async fn tree(
    State(state): State<AppState>,
    Query(params): Query<TreeParams>,
) -> ApiResult<Json<Value>> {
    let nodes = controller::tree(&state.container, params.uri.as_deref()).await?;
    Ok(Json(json!({
        "nodes": nodes.iter().map(node_json).collect::<Vec<_>>(),
    })))
}

/// `GET /api/sessions` — imported sessions.
pub async fn sessions(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let sessions = controller::sessions(&state.container).await?;
    Ok(Json(json!({ "sessions": sessions })))
}

/// `GET /api/stats` — store statistics.
pub async fn stats(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let stats = controller::stats(&state.container).await?;
    Ok(Json(json!({
        "total_items": stats.total_items,
        "items_by_kind": stats.items_by_kind,
        "total_sessions": stats.total_sessions,
        "total_nodes": stats.total_nodes,
        "nodes_by_kind": stats.nodes_by_kind,
        "data_dir": state.container.data_dir(),
    })))
}

// ── Namespaces ───────────────────────────────────────────────────────────────

/// `GET /api/namespaces` — namespaces with project counts.
pub async fn list_namespaces(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let namespaces = controller::list_namespaces(&state.container).await?;
    let list: Vec<Value> = namespaces
        .into_iter()
        .map(|(name, count)| json!({ "name": name, "project_count": count }))
        .collect();
    Ok(Json(json!({ "namespaces": list })))
}

#[derive(Debug, Deserialize)]
pub struct CreateNamespaceBody {
    pub name: String,
}

/// `POST /api/namespaces` — create a namespace.
pub async fn create_namespace(
    State(state): State<AppState>,
    Json(body): Json<CreateNamespaceBody>,
) -> ApiResult<Json<Value>> {
    let created = controller::create_namespace(&state.container, &body.name).await?;
    Ok(Json(json!({ "created": created })))
}

/// `DELETE /api/namespaces/{name}` — delete a namespace.
pub async fn delete_namespace(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<Json<Value>> {
    let deleted = controller::delete_namespace(&state.container, &name).await?;
    Ok(Json(json!({ "deleted": deleted })))
}

/// `GET /api/namespaces/{name}` — a namespace's member projects.
pub async fn show_namespace(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<Json<Value>> {
    let projects = controller::namespace_projects(&state.container, &name).await?;
    Ok(Json(json!({ "name": name, "projects": projects })))
}

#[derive(Debug, Deserialize)]
pub struct AssignProjectBody {
    pub project: String,
}

/// `POST /api/namespaces/{name}/projects` — add a project to a namespace.
pub async fn assign_project(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(body): Json<AssignProjectBody>,
) -> ApiResult<Json<Value>> {
    let assigned = controller::assign_project(&state.container, &name, &body.project).await?;
    Ok(Json(json!({ "assigned": assigned })))
}

/// `DELETE /api/namespaces/{name}/projects/{project}` — remove a project.
pub async fn unassign_project(
    State(state): State<AppState>,
    Path((name, project)): Path<(String, String)>,
) -> ApiResult<Json<Value>> {
    let removed = controller::unassign_project(&state.container, &name, &project).await?;
    Ok(Json(json!({ "removed": removed })))
}

// ── Commands ─────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ImportBody {
    pub path: String,
    #[serde(default)]
    pub force: bool,
}

/// `POST /api/import` — import a transcript file.
pub async fn import(
    State(state): State<AppState>,
    Json(body): Json<ImportBody>,
) -> ApiResult<Json<Value>> {
    use crate::application::ImportOutcome;
    let outcome = controller::import(&state.container, &body.path, body.force).await?;
    Ok(Json(match outcome {
        ImportOutcome::Imported { session, report } => json!({
            "imported": true,
            "session_id": session.id,
            "message_count": session.message_count,
            "operations_applied": report.applied.len(),
            "operations_skipped": report.skipped.len(),
        }),
        ImportOutcome::AlreadyImported { session } => json!({
            "imported": false,
            "already_imported": true,
            "session_id": session.id,
        }),
    }))
}

#[derive(Debug, Deserialize)]
pub struct AddResourceBody {
    pub source: String,
    #[serde(default)]
    pub name: Option<String>,
}

/// `POST /api/resources` — fetch, summarize, and store a resource.
pub async fn add_resource(
    State(state): State<AppState>,
    Json(body): Json<AddResourceBody>,
) -> ApiResult<Json<Value>> {
    let added =
        controller::add_resource(&state.container, &body.source, body.name.as_deref()).await?;
    Ok(Json(json!({
        "uri": added.node.uri(),
        "source": added.source,
        "chars": added.chars,
        "abstract": added.node.abstract_(),
    })))
}

// `POST /api/dream` used to run a cycle synchronously here. It now lives in
// `dream_routes`, which starts the cycle in the background and returns 202 — a
// full consolidation is many minutes of LLM calls, far too long to hold an HTTP
// connection open. The synchronous path is still the CLI's `memory-rs dream`,
// which goes through `controller::dream` directly.

// ── JSON helpers ─────────────────────────────────────────────────────────────

fn item_with_score((item, score): &(MemoryItem, f32)) -> Value {
    let mut value = serde_json::to_value(item).unwrap_or_default();
    if let Some(obj) = value.as_object_mut() {
        obj.insert("score".to_string(), json!(score));
    }
    value
}

/// A node as JSON, masking the internal manifest of Project digest nodes.
fn node_json(node: &MemoryNode) -> Value {
    let content = if controller::node_has_visible_content(node) {
        node.content()
    } else {
        ""
    };
    json!({
        "uri": node.uri(),
        "kind": node.kind().to_string(),
        "abstract": node.abstract_(),
        "overview": node.overview(),
        "content": content,
    })
}

fn parse_kind(kind: Option<&str>) -> ApiResult<Option<MemoryKind>> {
    match kind {
        None => Ok(None),
        Some(k) => MemoryKind::parse(k).map(Some).ok_or_else(|| {
            ApiError::from(crate::domain::DomainError::invalid_input(format!(
                "unknown memory kind '{k}'"
            )))
        }),
    }
}
