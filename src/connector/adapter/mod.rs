//! Memory-specific adapters.
//!
//! Implements the port traits defined in the application layer:
//! - [`DuckdbMemoryRepository`] — DuckDB-backed [`MemoryRepository`](crate::application::MemoryRepository)
//! - [`LocalSessionDiscovery`] — session discovery over Claude/OpenCode/Zed stores
//! - `parse_transcript_file` / `parse_transcript` — JSONL transcript parsing
//! - `fetch_resource` — URL/file fetch with HTML-to-Markdown cleaning
//! - `build_chat_client` / `build_embedding_client` — OpenAI client builders

mod duckdb_memory_repository;
mod embedding;
mod resource_fetch;
mod session_discovery;
mod transcript;

pub use duckdb_memory_repository::*;
pub use embedding::*;
pub use resource_fetch::*;
pub use session_discovery::*;
pub use transcript::*;
