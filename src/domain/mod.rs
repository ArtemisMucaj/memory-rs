//! Pure value types shared by every layer.
//!
//! No I/O, no async, no dependency on the storage or HTTP layers — only `serde`
//! and `thiserror`. Anything here can be constructed and asserted on in a test
//! without a database or a model server.

pub mod error;
pub mod memory;
pub mod session;
pub mod similarity;

pub use error::*;
pub use memory::*;
pub use session::*;
pub use similarity::*;
