//! Application entry surface: the dependency-injection container and the CLI
//! router. Depends on the adapters and the application use cases.

pub mod container;
pub mod router;

pub use container::{Container, ContainerConfig};
