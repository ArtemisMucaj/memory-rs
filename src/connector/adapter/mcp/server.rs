//! MCP server exposing the memory operations as tools.
//!
//! Every tool is a thin adapter over the shared
//! [`controller`](crate::connector::api::controller) layer — the same logic the
//! CLI and HTTP API use — so an assistant (over stdio) or a native app (over
//! HTTP) drives memory through one implementation.

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

use crate::connector::api::controller::{
    self, DeleteOutcome, SearchOutcome, SearchScope, ShowOutcome,
};
use crate::connector::api::Container;
use crate::domain::MemoryKind;

/// Server-side cap on how many results a single call returns.
const MAX_LIMIT: usize = 100;

fn default_limit() -> usize {
    10
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
    /// Restrict to a memory kind: preference / experience / skill / fact.
    #[serde(default)]
    pub kind: Option<String>,
    /// Scope to one project (its items + globals).
    #[serde(default)]
    pub project: Option<String>,
    /// Scope to a namespace (its member projects' items + globals). Mutually
    /// exclusive with `project`.
    #[serde(default)]
    pub namespace: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListInput {
    /// Restrict to a memory kind.
    #[serde(default)]
    pub kind: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ShowInput {
    /// A memory item id, a `kind/name` reference, or a `memory://` node URI.
    pub id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct TreeInput {
    /// Directory URI to list (e.g. `memory://sessions`); omit for the root.
    #[serde(default)]
    pub uri: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddResourceInput {
    /// A local file path or an http(s):// URL.
    pub source: String,
    /// Optional slug for the resource node; derived from the source otherwise.
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
    /// Recall long-term memories (preferences, experiences, skills, facts) by
    /// natural-language query, optionally scoped to a project or namespace.
    #[tool(name = "search_memory")]
    async fn search_memory(
        &self,
        params: Parameters<SearchInput>,
    ) -> Result<CallToolResult, McpError> {
        let input = params.0;
        let kind = parse_kind(input.kind.as_deref())?;
        let scope = match (input.namespace, input.project) {
            (Some(ns), _) => SearchScope::Namespace(ns),
            (None, Some(p)) => SearchScope::Project(p),
            (None, None) => SearchScope::All,
        };
        let limit = input.limit.min(MAX_LIMIT);

        let value = match controller::search(&self.container, &input.query, kind, &scope, limit)
            .await
            .map_err(internal)?
        {
            SearchOutcome::Hits(hits) => json!(hits
                .iter()
                .map(|(item, score)| json!({
                    "id": item.id(),
                    "kind": item.kind().as_str(),
                    "name": item.name(),
                    "content": item.content(),
                    "project": item.project(),
                    "score": score,
                }))
                .collect::<Vec<_>>()),
            SearchOutcome::EmptyNamespace(ns) => {
                json!({ "note": format!("namespace '{ns}' has no member projects"), "results": [] })
            }
        };
        ok_json(&value)
    }

    /// List stored memories, newest first, optionally filtered by kind. Use
    /// kind="preference" at session start to load all user preferences at once.
    #[tool(name = "list_memories")]
    async fn list_memories(
        &self,
        params: Parameters<ListInput>,
    ) -> Result<CallToolResult, McpError> {
        let kind = parse_kind(params.0.kind.as_deref())?;
        let items = controller::list_items(&self.container, kind)
            .await
            .map_err(internal)?;
        ok_json(&json!(items
            .iter()
            .map(|item| json!({
                "id": item.id(),
                "kind": item.kind().as_str(),
                "name": item.name(),
                "content": item.content(),
                "project": item.project(),
            }))
            .collect::<Vec<_>>()))
    }

    /// Read one memory item (by id or `kind/name`) or a `memory://` node (its
    /// L0/L1/L2 levels).
    #[tool(name = "read_memory")]
    async fn read_memory(&self, params: Parameters<ShowInput>) -> Result<CallToolResult, McpError> {
        let id = params.0.id;
        let value = match controller::show(&self.container, &id)
            .await
            .map_err(internal)?
        {
            ShowOutcome::Item(item) => json!({
                "type": "item",
                "id": item.id(),
                "kind": item.kind().as_str(),
                "name": item.name(),
                "content": item.content(),
                "project": item.project(),
            }),
            ShowOutcome::Node(node) => json!({
                "type": "node",
                "uri": node.uri(),
                "kind": node.kind().to_string(),
                "abstract": node.abstract_(),
                "overview": node.overview(),
                "content": if controller::node_has_visible_content(&node) { node.content() } else { "" },
            }),
            ShowOutcome::Many(items) => json!({
                "type": "ambiguous",
                "matches": items.iter().map(|i| json!({ "id": i.id(), "project": i.project() })).collect::<Vec<_>>(),
            }),
            ShowOutcome::NotFound => json!({ "type": "not_found", "id": id }),
        };
        ok_json(&value)
    }

    /// Browse the memory virtual filesystem: with no URI, the top-level roots;
    /// with a directory URI, its children and their one-line abstracts.
    #[tool(name = "browse_memory")]
    async fn browse_memory(
        &self,
        params: Parameters<TreeInput>,
    ) -> Result<CallToolResult, McpError> {
        let nodes = controller::tree(&self.container, params.0.uri.as_deref())
            .await
            .map_err(internal)?;
        ok_json(&json!(nodes
            .iter()
            .map(|n| json!({ "uri": n.uri(), "kind": n.kind().to_string(), "abstract": n.abstract_() }))
            .collect::<Vec<_>>()))
    }

    /// Delete a memory item by id (or unique `kind/name`).
    #[tool(name = "delete_memory")]
    async fn delete_memory(
        &self,
        params: Parameters<ShowInput>,
    ) -> Result<CallToolResult, McpError> {
        let id = params.0.id;
        let value = match controller::delete(&self.container, &id)
            .await
            .map_err(internal)?
        {
            DeleteOutcome::Deleted => json!({ "deleted": true }),
            DeleteOutcome::Ambiguous(items) => json!({
                "deleted": false,
                "reason": "ambiguous kind/name across projects; delete by id",
                "matches": items.iter().map(|i| json!({ "id": i.id(), "project": i.project() })).collect::<Vec<_>>(),
            }),
            DeleteOutcome::NotFound => json!({ "deleted": false, "reason": "not found" }),
        };
        ok_json(&value)
    }

    /// Add a resource (a file path or URL) to the memory virtual filesystem.
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
            "uri": added.node.uri(),
            "source": added.source,
            "chars": added.chars,
            "abstract": added.node.abstract_(),
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
                "Long-term memory server. Tools:\n\
                 • search_memory — recall memories by natural language (scope to a project or namespace)\n\
                 • list_memories — list memories, optionally by kind\n\
                 • read_memory — read one item or a memory:// node\n\
                 • browse_memory — browse the memory virtual filesystem\n\
                 • delete_memory — delete an item\n\
                 • add_resource — store a file/URL as a recallable resource\n\
                 • list_namespaces / create_namespace / assign_project — manage project groups"
                    .into(),
            ),
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn parse_kind(kind: Option<&str>) -> Result<Option<MemoryKind>, McpError> {
    match kind {
        None => Ok(None),
        Some(k) => MemoryKind::parse(k)
            .map(Some)
            .ok_or_else(|| McpError::invalid_params(format!("unknown memory kind '{k}'"), None)),
    }
}

fn internal(e: crate::domain::DomainError) -> McpError {
    McpError::internal_error(e.to_string(), None)
}

fn ok_json(value: &serde_json::Value) -> Result<CallToolResult, McpError> {
    let text = serde_json::to_string_pretty(value)
        .map_err(|e| McpError::internal_error(format!("serialize failed: {e}"), None))?;
    Ok(CallToolResult::success(vec![Content::text(text)]))
}
