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
use crate::application::{Recalled, DEFAULT_SESSION_LIMIT};
use crate::connector::api::controller::{
    self, ForgetOutcome, MemorySearchOutcome, MemoryShowOutcome, ResumeOutcome, SearchScope,
};
use crate::domain::Memory;
use std::collections::HashMap;

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
            "GET  /api/search?q=&project=&namespace=&limit=",
            "GET  /api/memory",
            "GET  /api/memory/{id}  -> {type: memory, memory} | {type: resource, resource}",
            "DELETE /api/memory/{id}  -> {deleted: true}",
            "GET  /api/entities",
            "GET  /api/entities/{id}",
            "GET  /api/tree?uri=",
            "GET  /api/sessions",
            "GET  /api/resume?project=&namespace=&limit=",
            "GET  /api/sessions/discover",
            "GET  /api/sessions/transcript?source=&id=",
            "POST /api/sessions/import  {source, id, force?}",
            "GET  /api/sessions/import",
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
    let scope = match (params.namespace, params.project) {
        (Some(ns), _) => SearchScope::Namespace(ns),
        (None, Some(p)) => SearchScope::Project(p),
        (None, None) => SearchScope::All,
    };
    let limit = params.limit.unwrap_or(10).min(100);

    match controller::recall_memories(&state.container, &params.q, None, &scope, limit).await? {
        MemorySearchOutcome::Hits(hits) => {
            let found: Vec<Memory> = hits.iter().map(|h| h.memory.clone()).collect();
            let labels = controller::entity_labels(&state.container, &found).await?;
            Ok(Json(json!({
                "results": hits
                    .iter()
                    .map(|h| memory_with_score(h, &labels))
                    .collect::<Vec<_>>(),
            })))
        }
        MemorySearchOutcome::EmptyNamespace(ns) => Ok(Json(json!({
            "results": [],
            "note": format!("namespace '{ns}' has no member projects"),
        }))),
    }
}

#[derive(Debug, Deserialize)]
pub struct ListParams {
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub namespace: Option<String>,
}

/// `GET /api/memory` — list memories, newest first.
pub async fn list_memories(
    State(state): State<AppState>,
    Query(params): Query<ListParams>,
) -> ApiResult<Json<Value>> {
    let scope = match (params.namespace, params.project) {
        (Some(ns), _) => SearchScope::Namespace(ns),
        (None, Some(p)) => SearchScope::Project(p),
        (None, None) => SearchScope::All,
    };
    let projects = match controller::resolve_scope(&state.container, &scope).await? {
        controller::ScopeResolution::All => None,
        controller::ScopeResolution::Projects(p) => Some(p),
        controller::ScopeResolution::EmptyNamespace(ns) => {
            return Ok(Json(json!({
                "memories": [],
                "note": format!("namespace '{ns}' has no member projects"),
            })));
        }
    };
    let memories = controller::list_memories(&state.container, projects.as_deref()).await?;
    let labels = controller::entity_labels(&state.container, &memories).await?;
    let rendered: Vec<Value> = memories
        .iter()
        .map(|m| memory_json(m, &labels, None))
        .collect();
    Ok(Json(json!({ "memories": rendered })))
}

/// `GET /api/memory/{id}` — show a memory, or a resource URI.
pub async fn show(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    match controller::show_memory(&state.container, &id).await? {
        MemoryShowOutcome::Resource(resource) => Ok(Json(json!({
            "type": "resource",
            "resource": {
                "uri": resource.uri,
                "source": resource.source,
                "name": resource.name,
                "abstract": resource.abstract_,
                "overview": resource.overview,
                "content": resource.content,
                "created_at": resource.created_at,
            }
        }))),
        MemoryShowOutcome::Memory(memory) => {
            let labels = controller::entity_labels(&state.container, &[*memory.clone()]).await?;
            Ok(Json(json!({
                "type": "memory",
                "memory": memory_json(&memory, &labels, None),
            })))
        }
        MemoryShowOutcome::NotFound => Err(ApiError::from(crate::domain::DomainError::not_found(
            format!("no memory matches '{id}'"),
        ))),
    }
}

/// `DELETE /api/memory/{id}` — hard-delete a memory.
pub async fn delete(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    match controller::forget_memory(&state.container, &id).await? {
        ForgetOutcome::Deleted => Ok(Json(json!({ "deleted": true }))),
        ForgetOutcome::NotFound => Err(ApiError::from(crate::domain::DomainError::not_found(
            format!("no memory '{id}'"),
        ))),
    }
}

#[derive(Debug, Deserialize)]
pub struct TreeParams {
    #[serde(default)]
    pub uri: Option<String>,
}

/// `GET /api/entities` — the resolved entities memories are anchored to.
pub async fn entities(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let entities = controller::list_entities(&state.container).await?;
    let list: Vec<Value> = entities.iter().map(entity_json).collect();
    Ok(Json(json!({ "entities": list })))
}

/// `GET /api/entities/{id}` — one entity with the memories referencing it.
pub async fn entity(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    let Some((entity, memories)) = controller::show_entity(&state.container, &id).await? else {
        return Err(ApiError::from(crate::domain::DomainError::not_found(
            format!("no entity '{id}'"),
        )));
    };
    let labels = controller::entity_labels(&state.container, &memories).await?;
    Ok(Json(json!({
        "entity": {
            "id": entity.id,
            "canonical_name": entity.canonical_name,
            "entity_type": entity.entity_type,
            "names": entity.names,
            "memory_count": memories.len(),
            "created_at": entity.created_at,
            "updated_at": entity.updated_at,
        },
        "memories": memories.iter().map(|m| memory_json(m, &labels, None)).collect::<Vec<_>>(),
    })))
}

fn entity_json(summary: &controller::EntitySummary) -> Value {
    json!({
        "id": summary.entity.id,
        "canonical_name": summary.entity.canonical_name,
        "entity_type": summary.entity.entity_type,
        "names": summary.entity.names,
        "memory_count": summary.memory_count,
    })
}

/// `GET /api/tree` — list sessions and resources (the old L0/L1/L2 tree is
/// gone; this is the minimum that still answers "what is in here").
pub async fn tree(
    State(state): State<AppState>,
    Query(params): Query<TreeParams>,
) -> ApiResult<Json<Value>> {
    let nodes = controller::tree(&state.container, params.uri.as_deref()).await?;
    Ok(Json(json!({ "nodes": nodes })))
}

/// `GET /api/sessions` — imported sessions.
pub async fn sessions(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let sessions = controller::sessions(&state.container).await?;
    Ok(Json(json!({ "sessions": sessions })))
}

#[derive(Debug, Deserialize)]
pub struct ResumeParams {
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

/// `GET /api/resume` — recent work in scope: the latest sessions, what each
/// was about, and the memories it produced.
pub async fn resume(
    State(state): State<AppState>,
    Query(params): Query<ResumeParams>,
) -> ApiResult<Json<Value>> {
    let scope = match (params.namespace, params.project) {
        (Some(ns), _) => SearchScope::Namespace(ns),
        (None, Some(p)) => SearchScope::Project(p),
        (None, None) => SearchScope::All,
    };
    let limit = params.limit.unwrap_or(DEFAULT_SESSION_LIMIT);

    match controller::resume(&state.container, &scope, limit).await? {
        ResumeOutcome::Briefing(briefing) => {
            let mut sessions = Vec::with_capacity(briefing.sessions.len());
            for recap in &briefing.sessions {
                let labels = controller::entity_labels(&state.container, &recap.memories).await?;
                sessions.push(json!({
                    "id": recap.session.id,
                    "source": recap.session.source,
                    "project": recap.session.project,
                    "imported_at": recap.session.imported_at,
                    "message_count": recap.session.message_count,
                    "memories": recap
                        .memories
                        .iter()
                        .map(|m| memory_json(m, &labels, None))
                        .collect::<Vec<_>>(),
                }));
            }
            Ok(Json(json!({
                "projects": briefing.projects,
                "more": briefing.more,
                "sessions": sessions,
            })))
        }
        ResumeOutcome::EmptyNamespace(ns) => Ok(Json(json!({
            "sessions": [],
            "note": format!("namespace '{ns}' has no member projects"),
        }))),
    }
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
            "memories_written": report.memories_written,
            "memories_deduped": report.memories_deduped,
            "entities_created": report.entities_created,
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
        "uri": added.resource.uri,
        "source": added.source,
        "chars": added.chars,
        "abstract": added.resource.abstract_,
    })))
}

// ── JSON helpers ─────────────────────────────────────────────────────────────

/// A search hit with its fused score.
fn memory_with_score(hit: &Recalled, labels: &HashMap<String, String>) -> Value {
    memory_json(&hit.memory, labels, Some(hit.score))
}

/// A memory as JSON, with its subject/object entity ids resolved to their
/// canonical names.
fn memory_json(memory: &Memory, labels: &HashMap<String, String>, score: Option<f32>) -> Value {
    let mut value = serde_json::to_value(memory).unwrap_or_default();
    if let Some(obj) = value.as_object_mut() {
        if let Some(label) = controller::entity_ref_label(&memory.subject, labels) {
            obj.insert("subject_label".to_string(), json!(label));
        }
        if let Some(label) = controller::entity_ref_label(&memory.object, labels) {
            obj.insert("object_label".to_string(), json!(label));
        }
        if let Some(score) = score {
            obj.insert("score".to_string(), json!(score));
        }
    }
    value
}
