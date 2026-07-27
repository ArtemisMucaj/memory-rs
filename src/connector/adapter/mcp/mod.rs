//! MCP server (the `mcp` command, and mounted on the HTTP `serve` process).
//!
//! Exposes the memory operations as MCP tools over stdio (for a direct
//! assistant integration) or streamable HTTP (mounted at `/mcp` by the
//! management server). Both use the same [`MemoryMcpServer`], whose tools call
//! the shared [`controller`](crate::connector::api::controller) layer.

mod server;

use std::sync::Arc;

use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::StreamableHttpService;
use rmcp::ServiceExt;

pub use server::MemoryMcpServer;

use crate::connector::api::Container;
use crate::domain::DomainError;

/// Serve the MCP protocol over stdio until the client disconnects. The stdio
/// transport owns stdin/stdout, so this must be the process's sole use of them
/// (logs go to stderr / the capture, never stdout).
pub async fn serve_stdio(container: Container) -> Result<(), DomainError> {
    let server = MemoryMcpServer::new(Arc::new(container));
    let running = server
        .serve(rmcp::transport::stdio())
        .await
        .map_err(|e| DomainError::internal(format!("failed to start MCP stdio server: {e}")))?;
    running
        .waiting()
        .await
        .map_err(|e| DomainError::internal(format!("MCP stdio server error: {e}")))?;
    Ok(())
}

/// Build a streamable-HTTP MCP service the management server mounts at `/mcp`,
/// so one `serve` process offers both the REST API and MCP over HTTP.
pub fn http_service(container: Arc<Container>) -> StreamableHttpService<MemoryMcpServer> {
    StreamableHttpService::new(
        move || Ok(MemoryMcpServer::new(container.clone())),
        Arc::new(LocalSessionManager::default()),
        Default::default(),
    )
}
