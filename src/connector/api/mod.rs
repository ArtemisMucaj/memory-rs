//! Application entry surface: the dependency-injection container, the shared
//! controllers, and the CLI router. Depends on the adapters and the application
//! use cases.

pub mod container;
pub mod controller;
pub mod router;

pub use container::{Container, ContainerConfig};
