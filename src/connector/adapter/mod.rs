//! Memory-specific adapters.
//!
//! Implements the port traits defined in the application layer:
//! - [`DuckdbStore`] — DuckDB-backed [`NodeRepository`](crate::application::NodeRepository)
//! - [`LocalSessionDiscovery`] — session discovery over Claude/OpenCode/Zed stores
//! - `parse_transcript_file` / `parse_transcript` — JSONL transcript parsing
//! - `fetch_resource` — URL/file fetch with HTML-to-Markdown cleaning
//! - `build_chat_client` / `build_embedding_client` — OpenAI client builders
//! - [`MemoryConfig`] — on-disk `config.json` (endpoints, embedding, dream)
//! - `copilot` — GitHub Copilot as an alternative chat backend

mod config;
pub mod copilot;
/// `impl MemoryRepository for DuckdbStore` — no items of its own to
/// re-export, so the module is declared but not glob-imported.
mod duckdb_memory_repository;
mod duckdb_store;
mod embedding;
pub mod management;
pub mod mcp;
mod resource_fetch;
mod session_discovery;
mod transcript;

pub use config::*;
pub use duckdb_store::*;
pub use embedding::*;
pub use resource_fetch::*;
pub use session_discovery::*;
pub use transcript::*;
