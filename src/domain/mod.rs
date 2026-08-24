//! Pure value types shared by every layer.
//!
//! No I/O, no async, no dependency on the storage or HTTP layers — only
//! `serde` and `thiserror`. Anything here can be constructed and asserted on
//! in a test without a database or a model server.
//!
//! [`memory_graph`] holds the store-facing model: [`Memory`] (a
//! self-contained statement plus the entities it mentions) and [`Entity`]
//! (the anchor a memory points at). [`memory`] holds the [`MemoryKind`]
//! vocabulary and the session transcript types.

pub mod error;
pub mod memory;
pub mod memory_graph;
pub mod resource;
pub mod session;
pub mod similarity;

pub use error::*;
pub use memory::*;
pub use memory_graph::*;
pub use resource::*;
pub use session::*;
pub use similarity::*;
