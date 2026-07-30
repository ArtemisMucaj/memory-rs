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
use crate::application::{MemoryRef, Recalled, DEFAULT_SESSION_LIMIT};
use crate::connector::api::controller::{
    self, ForgetOutcome, MemorySearchOutcome, MemoryShowOutcome, ResumeOutcome, SearchScope,
};
use crate::domain::{Memory, MemoryKind, MemoryNode, MemoryStatus};
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
            "GET  /api/search?q=&kind=&project=&namespace=&limit=",
            "GET  /api/memory?kind=&status=  (status: active|superseded|retracted|all; default active)",
            "GET  /api/memory/{id}  -> {type: memory, memory, edges} | {type: node, node}",
            "DELETE /api/memory/{id}  -> {retracted: true}",
            "GET  /api/conflicts",
            "GET  /api/entities",
            "GET  /api/entities/{id}",
            "GET  /api/tree?uri=",
            "GET  /api/sessions",
            "GET  /api/resume?project=&namespace=&limit=",
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

    match controller::recall_memories(&state.container, &params.q, kind, &scope, limit).await? {
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
    pub kind: Option<String>,
    /// Lifecycle status. Defaults to `active`; pass `all` for the full log
    /// including history. Unresolved conflicts live at `GET /api/conflicts`,
    /// not behind a status — both sides of a disagreement stay active.
    #[serde(default)]
    pub status: Option<String>,
}

/// `GET /api/memory` — list memories, optionally filtered by kind and status.
///
/// Defaults to `active` rather than every status: the log holds superseded and
/// retracted memories forever, and a client that rendered the raw list would show
/// a user things the store has already decided are not true.
pub async fn list_memories(
    State(state): State<AppState>,
    Query(params): Query<ListParams>,
) -> ApiResult<Json<Value>> {
    let kind = parse_kind(params.kind.as_deref())?;
    let status = parse_status(params.status.as_deref())?;
    let memories = controller::list_memories(&state.container, kind, status).await?;
    let labels = controller::entity_labels(&state.container, &memories).await?;
    let rendered: Vec<Value> = memories
        .iter()
        .map(|m| memory_json(m, &labels, None))
        .collect();
    Ok(Json(json!({ "memories": rendered })))
}

/// `GET /api/memory/{id}` — show a memory with its edges, or a node URI.
pub async fn show(State(state): State<AppState>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    match controller::show_memory(&state.container, &id).await? {
        MemoryShowOutcome::Node(node) => {
            Ok(Json(json!({ "type": "node", "node": node_json(&node) })))
        }
        MemoryShowOutcome::Memory { memory, edges } => {
            let labels = controller::entity_labels(&state.container, &[*memory.clone()]).await?;
            Ok(Json(json!({
                "type": "memory",
                "memory": memory_json(&memory, &labels, None),
                "edges": edges,
            })))
        }
        MemoryShowOutcome::NotFound => Err(ApiError::from(crate::domain::DomainError::not_found(
            format!("no memory matches '{id}'"),
        ))),
    }
}

/// `DELETE /api/memory/{id}` — retract a memory.
///
/// The response says `retracted`, not `deleted`, because nothing is removed:
/// the memory stays in the log with `status = retracted` and simply stops
/// answering queries. A client that showed "deleted" would be lying about what
/// it just did.
pub async fn delete(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    match controller::forget_memory(&state.container, &id).await? {
        ForgetOutcome::Retracted => Ok(Json(json!({ "retracted": true }))),
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
            // Same shape as the list endpoint, deliberately. A client decoding
            // both into one type would otherwise fail on the detail response
            // for a field it never needed — and a lenient client would just
            // render an empty pane instead.
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
        // Every name the entity answers to, canonical included. Under "known
        // as" that reads correctly, and it is also the honest picture of what
        // the lookup index holds.
        "names": summary.entity.names,
        "memory_count": summary.memory_count,
    })
}

/// `GET /api/conflicts` — unresolved disagreements, both sides still active.
pub async fn conflicts(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let conflicts = controller::memory_conflicts(&state.container).await?;
    let list: Vec<Value> = conflicts
        .iter()
        .map(|c| json!({ "recorded_at": c.recorded_at, "a": c.a, "b": c.b }))
        .collect();
    Ok(Json(json!({ "conflicts": list })))
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

#[derive(Debug, Deserialize)]
pub struct ResumeParams {
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

/// `GET /api/resume` — recent work in scope: the latest sessions, what each was
/// about, and the memories it produced.
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
                    "summary": recap.summary,
                    "overview": recap.overview,
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

/// `GET /api/stats` — store statistics.
pub async fn stats(State(state): State<AppState>) -> ApiResult<Json<Value>> {
    let stats = controller::stats(&state.container).await?;
    let memories = controller::memory_stats(&state.container).await?;
    Ok(Json(json!({
        "total_memories": memories.total_memories,
        "memories_by_kind": memories.memories_by_kind,
        // Everything outside `active` is history: superseded links and
        // retractions. Unresolved conflicts are not a status — see
        // `GET /api/conflicts`.
        "memories_by_status": memories.memories_by_status,
        "total_entities": memories.total_entities,
        "total_edges": memories.total_edges,
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
            "memories_written": report.memories_written,
            "memories_corroborated": report.memories_corroborated,
            "memories_superseded": report.memories_superseded,
            // Both sides of a contradiction stay recallable, so this is not a
            // "something went missing" counter — it is how much disagreement
            // this session introduced for consolidation to reconcile.
            "conflicts_recorded": report.conflicts_recorded,
            "entities_created": report.entities_created,
            "edges_added": report.edges_added,
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

/// A search hit with a **compact** provenance summary.
///
/// Search returns many rows, and a full supersession chain per row would bury
/// the answers in their own history. Counts plus the live disagreements are
/// enough for a client to badge a result; `GET /api/memory/{id}` carries the
/// full path for the one the user then opens.
fn memory_with_score(hit: &Recalled, labels: &HashMap<String, String>) -> Value {
    let mut value = memory_json(&hit.memory, labels, Some(hit.score));
    if let Some(obj) = value.as_object_mut() {
        obj.insert(
            "provenance".to_string(),
            json!({
                "supersedes_count": hit.provenance.supersedes.len(),
                "chain_truncated": hit.provenance.chain_truncated,
                "corroborations": hit.provenance.corroborations,
                "contradicted_by": hit.provenance.contradicted_by
                    .iter().map(memory_ref_json).collect::<Vec<_>>(),
                "refinements_count": hit.provenance.refinements.len(),
            }),
        );
    }
    value
}

/// A memory as JSON, with its subject/object entity ids resolved to their
/// canonical names.
///
/// The raw `subject`/`object` fields are left exactly as the domain has them —
/// a client that wants the id still gets it — and the readable form is added
/// alongside. Replacing them would strip information a caller may need to
/// follow the graph.
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

fn memory_ref_json(r: &MemoryRef) -> Value {
    json!({ "id": r.id, "statement": r.statement, "recorded_at": r.recorded_at })
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

/// Parse the `status` filter. Absent means `active`; the explicit string `all`
/// is the only way to see the whole log, so history has to be asked for.
fn parse_status(status: Option<&str>) -> ApiResult<Option<MemoryStatus>> {
    match status {
        None => Ok(Some(MemoryStatus::Active)),
        Some(s) if s.eq_ignore_ascii_case("all") => Ok(None),
        Some(s) => MemoryStatus::parse(s).map(Some).ok_or_else(|| {
            ApiError::from(crate::domain::DomainError::invalid_input(format!(
                "unknown memory status '{s}' (expected active, superseded, retracted, or all)"
            )))
        }),
    }
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
