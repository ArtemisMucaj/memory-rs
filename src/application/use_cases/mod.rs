//! Use cases — the orchestration logic of the memory system.
//!
//! Each file is one use case; the `*_prompt` modules hold the LLM prompt
//! construction for their sibling use case and are internal.
//!
//! [`llm_json`] and [`memory_support`] are not use cases: they are the shared
//! primitives the use cases call. They export nothing public, so they are
//! reached by path (`use_cases::llm_json::…`) rather than glob re-exported —
//! a `pub use` of a wholly `pub(crate)` module re-exports nothing and warns.

mod import_session;
pub(crate) mod llm_json;
mod memory_browse;
mod memory_dream;
mod memory_dream_prompt;
mod memory_extraction;
mod memory_extraction_prompt;
mod memory_ingestion;
mod memory_ingestion_prompt;
mod memory_recall;
mod memory_search;
mod memory_summary;
pub(crate) mod memory_support;

pub use import_session::*;
pub use memory_browse::*;
pub use memory_dream::*;
pub use memory_extraction::*;
pub use memory_ingestion::*;
pub use memory_recall::*;
pub use memory_search::*;
pub use memory_summary::*;
