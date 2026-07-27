//! Concrete adapters — DuckDB store, session discovery, transcript parsing,
//! resource fetch, and OpenAI-compatible client builders.
//!
//! Depends on the application layer (port traits) and the domain layer.

pub mod adapter;
pub mod api;

pub use adapter::*;
