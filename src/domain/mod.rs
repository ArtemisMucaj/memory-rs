//! Pure value types shared by every layer.
//!
//! No I/O, no async, no dependency on the storage or HTTP layers — only `serde`
//! and `thiserror`. Anything here can be constructed and asserted on in a test
//! without a database or a model server.
//!
//! [`memory_graph`] holds the append-only model a stored memory actually lives
//! in — [`Memory`], its typed [`MemoryEdge`]s and the [`Entity`]s they anchor
//! to. [`memory`] holds the older flat [`MemoryItem`], the vocabulary both
//! share ([`MemoryKind`]) and the session transcript types.

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
