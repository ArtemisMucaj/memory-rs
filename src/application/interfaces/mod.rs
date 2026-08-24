//! Port traits — the abstractions the use cases depend on, implemented by the
//! connector layer.
//!
//! The LLM and embedding ports are *not* defined here: use cases consume
//! [`openai_rs::ChatClient`] and [`openai_rs::EmbeddingClient`] directly.

mod embedder;
mod memory_repository;
mod session_discovery;

pub use embedder::*;
pub use memory_repository::*;
pub use session_discovery::*;
