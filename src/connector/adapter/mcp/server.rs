//! MCP server exposing the memory operations as tools.
//!
//! Every tool is a thin adapter over the shared
//! [`controller`](crate::connector::api::controller) layer — the same logic
//! the CLI and HTTP API use — so an assistant (over stdio) or a native app
//! (over HTTP) drives memory through one implementation.
//!
//! Tool names are stable across the simplification: external clients depend
//! on them. Some parameters that no longer have meaning (`kind`, `status`)
//! are accepted and ignored.

use std::sync::Arc;

use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolResult, Content, Implementation, ProtocolVersion, ServerCapabilities, ServerInfo,
};
use rmcp::{tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use crate::application::DEFAULT_SESSION_LIMIT;
use crate::connector::api::controller::{
    self, ForgetOutcome, MemorySearchOutcome, MemoryShowOutcome, ResumeOutcome, SearchScope,
};
use crate::connector::api::Container;
use crate::domain::Memory;

/// Server-side cap on how many results a single call returns.
const MAX_LIMIT: usize = 100;

fn default_limit() -> usize {
    10
}

fn default_session_limit() -> usize {
    DEFAULT_SESSION_LIMIT
}

/// One scope, built the same way for every tool that takes one. `project`
/// and `namespace` are mutually exclusive; passing both is a caller error
/// rather than a silent preference for one.
fn scope_from(project: Option<String>, namespace: Option<String>) -> Result<SearchScope, McpError> {
    match (namespace, project) {
        (Some(_), Some(_)) => Err(McpError::invalid_params(
            "pass either `project` or `namespace`, not both",
            None,
        )),
        (Some(ns), None) => Ok(SearchScope::Namespace(ns)),
        (None, Some(p)) => Ok(SearchScope::Project(p)),
        (None, None) => Ok(SearchScope::All),
    }
}

/// The MCP server: holds the shared container.
#[derive(Clone)]
pub struct MemoryMcpServer {
    container: Arc<Container>,
    tool_router: ToolRouter<Self>,
}

impl MemoryMcpServer {
    pub fn new(container: Arc<Container>) -> Self {
        Self {
            container,
            tool_router: Self::tool_router(),
        }
    }
}

// ── Tool inputs ──────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchInput {
    /// Natural-language query.
    pub query: String,
    /// Max results (default 10, server cap 100).
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Accepted for backward compatibility; only `fact` exists now.
    #[serde(default)]
    #[allow(dead_code)]
    pub kind: Option<String>,
    /// Scope to one project (its items + globals).
    #[serde(default)]
    pub project: Option<String>,
    /// Scope to a namespace (its member projects' items + globals).
    /// Mutually exclusive with `project`.
    #[serde(default)]
    pub namespace: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListInput {
    /// Accepted for backward compatibility; ignored (only `fact` exists).
    #[serde(default)]
    #[allow(dead_code)]
    pub kind: Option<String>,
    /// Accepted for backward compatibility; ignored (no lifecycle status).
    #[serde(default)]
    #[allow(dead_code)]
    pub status: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ResumeInput {
    /// Scope to the project you are about to work in (its sessions only).
    #[serde(default)]
    pub project: Option<String>,
    /// Scope to a namespace — sessions across its member projects. Mutually
    /// exclusive with `project`.
    #[serde(default)]
    pub namespace: Option<String>,
    /// How many sessions to cover, newest first (default 5, server cap 50).
    #[serde(default = "default_session_limit")]
    pub limit: usize,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ShowInput {
    /// A memory id, or a `memory://resources/<name>` URI.
    pub id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct TreeInput {
    /// Optional subtree: `memory://sessions` or `memory://resources`. Omit
    /// for the root.
    #[serde(default)]
    pub uri: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddResourceInput {
    /// A local file path or an http(s):// URL.
    pub source: String,
    /// Optional slug for the resource; derived from the source otherwise.
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct NamespaceInput {
    /// Namespace name.
    pub name: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AssignInput {
    /// Namespace name.
    pub namespace: String,
    /// Project (git owner/repo) to add.
    pub project: String,
}

// ── Tools ────────────────────────────────────────────────────────────────────

#[tool_router]
impl MemoryMcpServer {
    /// Recall long-term memory by natural-language query, optionally scoped
    /// to a project or namespace. Returns atomic facts, best match first.
    /// Ranking is RRF over semantic similarity, keyword match, and recency —
    /// newer memories carry more weight.
    #[tool(name = "search_memories")]
    async fn search_memories(
        &self,
        params: Parameters<SearchInput>,
    ) -> Result<CallToolResult, McpError> {
        let input = params.0;
        let scope = scope_from(input.project, input.namespace)?;
        let limit = input.limit.min(MAX_LIMIT);

        let value = match controller::recall_memories(
            &self.container,
            &input.query,
            None,
            &scope,
            limit,
        )
        .await
        .map_err(internal)?
        {
            MemorySearchOutcome::Hits(hits) => {
                let found: Vec<Memory> = hits.iter().map(|h| h.memory.clone()).collect();
                let labels = controller::entity_labels(&self.container, &found)
                    .await
                    .map_err(internal)?;
                json!(hits
                    .iter()
                    .map(|hit| memory_json(&hit.memory, &labels, Some(hit.score)))
                    .collect::<Vec<_>>())
            }
            MemorySearchOutcome::EmptyNamespace(ns) => {
                json!({ "note": format!("namespace '{ns}' has no member projects"), "results": [] })
            }
        };
        ok_json(&value)
    }

    /// List stored memories, newest first. Use at session start to load
    /// everything currently on record.
    #[tool(name = "list_memories")]
    async fn list_memories(
        &self,
        _params: Parameters<ListInput>,
    ) -> Result<CallToolResult, McpError> {
        let memories = controller::list_memories(&self.container, None)
            .await
            .map_err(internal)?;
        let labels = controller::entity_labels(&self.container, &memories)
            .await
            .map_err(internal)?;
        ok_json(&json!(memories
            .iter()
            .map(|memory| memory_json(memory, &labels, None))
            .collect::<Vec<_>>()))
    }

    /// Catch up on recent work before doing anything else in a project: the
    /// latest sessions and the durable memories each produced. Call this at
    /// the start of a session so the user does not have to re-explain what
    /// they were doing. Scope it with `project` (or `namespace`) to the work
    /// you are about to continue.
    #[tool(name = "resume_work")]
    async fn resume_work(
        &self,
        params: Parameters<ResumeInput>,
    ) -> Result<CallToolResult, McpError> {
        let input = params.0;
        let scope = scope_from(input.project, input.namespace)?;
        let briefing = match controller::resume(&self.container, &scope, input.limit)
            .await
            .map_err(internal)?
        {
            ResumeOutcome::Briefing(b) => b,
            ResumeOutcome::EmptyNamespace(ns) => {
                return ok_json(&json!({ "empty_namespace": ns, "sessions": [] }));
            }
        };

        let mut sessions = Vec::with_capacity(briefing.sessions.len());
        for recap in &briefing.sessions {
            let labels = controller::entity_labels(&self.container, &recap.memories)
                .await
                .map_err(internal)?;
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
        ok_json(&json!({
            "projects": briefing.projects,
            "more": briefing.more,
            "sessions": sessions,
        }))
    }

    /// Read one memory by id, or a `memory://resources/<name>` resource.
    #[tool(name = "read_memory")]
    async fn read_memory(&self, params: Parameters<ShowInput>) -> Result<CallToolResult, McpError> {
        let id = params.0.id;
        let value = match controller::show_memory(&self.container, &id)
            .await
            .map_err(internal)?
        {
            MemoryShowOutcome::Memory(memory) => {
                let labels = controller::entity_labels(&self.container, &[*memory.clone()])
                    .await
                    .map_err(internal)?;
                let mut value = memory_json(&memory, &labels, None);
                if let Some(obj) = value.as_object_mut() {
                    obj.insert("type".to_string(), json!("memory"));
                }
                value
            }
            MemoryShowOutcome::Resource(resource) => json!({
                "type": "resource",
                "uri": resource.uri,
                "source": resource.source,
                "name": resource.name,
                "abstract": resource.abstract_,
                "overview": resource.overview,
                "content": resource.content,
            }),
            MemoryShowOutcome::NotFound => json!({ "type": "not_found", "id": id }),
        };
        ok_json(&value)
    }

    /// List what is in the store: resources at the root, sessions under
    /// `memory://sessions`.
    #[tool(name = "browse_memory")]
    async fn browse_memory(
        &self,
        params: Parameters<TreeInput>,
    ) -> Result<CallToolResult, McpError> {
        let nodes = controller::tree(&self.container, params.0.uri.as_deref())
            .await
            .map_err(internal)?;
        ok_json(&json!(nodes))
    }

    /// Forget a memory by id. Hard delete: the row is gone.
    #[tool(name = "forget_memory")]
    async fn forget_memory(
        &self,
        params: Parameters<ShowInput>,
    ) -> Result<CallToolResult, McpError> {
        let id = params.0.id;
        let value = match controller::forget_memory(&self.container, &id)
            .await
            .map_err(internal)?
        {
            ForgetOutcome::Deleted => json!({ "deleted": true, "id": id }),
            ForgetOutcome::NotFound => json!({ "deleted": false, "reason": "not found" }),
        };
        ok_json(&value)
    }

    /// Add a resource (a file path or URL) to the store.
    #[tool(name = "add_resource")]
    async fn add_resource(
        &self,
        params: Parameters<AddResourceInput>,
    ) -> Result<CallToolResult, McpError> {
        let input = params.0;
        let added = controller::add_resource(&self.container, &input.source, input.name.as_deref())
            .await
            .map_err(internal)?;
        ok_json(&json!({
            "uri": added.resource.uri,
            "source": added.source,
            "chars": added.chars,
            "abstract": added.resource.abstract_,
        }))
    }

    /// List namespaces (groups of projects) with their project counts.
    #[tool(name = "list_namespaces")]
    async fn list_namespaces(&self) -> Result<CallToolResult, McpError> {
        let namespaces = controller::list_namespaces(&self.container)
            .await
            .map_err(internal)?;
        ok_json(&json!(namespaces
            .iter()
            .map(|(name, count)| json!({ "name": name, "project_count": count }))
            .collect::<Vec<_>>()))
    }

    /// Create a namespace.
    #[tool(name = "create_namespace")]
    async fn create_namespace(
        &self,
        params: Parameters<NamespaceInput>,
    ) -> Result<CallToolResult, McpError> {
        let created = controller::create_namespace(&self.container, &params.0.name)
            .await
            .map_err(internal)?;
        ok_json(&json!({ "created": created }))
    }

    /// Add a project (git owner/repo) to a namespace.
    #[tool(name = "assign_project")]
    async fn assign_project(
        &self,
        params: Parameters<AssignInput>,
    ) -> Result<CallToolResult, McpError> {
        let input = params.0;
        let assigned =
            controller::assign_project(&self.container, &input.namespace, &input.project)
                .await
                .map_err(internal)?;
        ok_json(&json!({ "assigned": assigned }))
    }
}

#[tool_handler]
impl ServerHandler for MemoryMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation::from_build_env(),
            instructions: Some(
                "Long-term memory server. Memory is stored as atomic facts (subject–predicate–object) \
                 anchored to resolved entities. Recall ranks by RRF over semantic similarity, keyword \
                 match, and recency — newer memories carry more weight.\n\
                 Tools:\n\
                 • resume_work — catch up on recent sessions before starting work in a project\n\
                 • search_memories — recall memories by natural language (scope to a project or namespace)\n\
                 • list_memories — list every memory, newest first\n\
                 • read_memory — read one memory, or a memory://resources/<name> resource\n\
                 • browse_memory — list what is in the store\n\
                 • forget_memory — hard-delete a memory\n\
                 • add_resource — store a file/URL for later recall\n\
                 • list_namespaces / create_namespace / assign_project — manage project groups"
                    .into(),
            ),
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// One memory as the wire shape every tool here returns.
fn memory_json(
    memory: &Memory,
    labels: &std::collections::HashMap<String, String>,
    score: Option<f32>,
) -> serde_json::Value {
    let mut value = json!({
        "id": memory.id,
        "kind": memory.kind.as_str(),
        "statement": memory.statement,
        // The readable subject — an entity id would be meaningless to a
        // model deciding whether this memory answers the question.
        "subject": controller::entity_ref_label(&memory.subject, labels),
        "predicate": memory.predicate.as_str(),
        "object": controller::entity_ref_label(&memory.object, labels),
        "project": memory.project,
        "source_kind": memory.source_kind.as_str(),
        "confidence": memory.confidence,
        "recorded_at": memory.recorded_at,
    });
    if let (Some(score), Some(obj)) = (score, value.as_object_mut()) {
        obj.insert("score".to_string(), json!(score));
    }
    value
}

fn internal(e: crate::domain::DomainError) -> McpError {
    McpError::internal_error(e.to_string(), None)
}

fn ok_json(value: &serde_json::Value) -> Result<CallToolResult, McpError> {
    let text = serde_json::to_string_pretty(value)
        .map_err(|e| McpError::internal_error(format!("serialize failed: {e}"), None))?;
    Ok(CallToolResult::success(vec![Content::text(text)]))
}
