//! HTTP management API (the `serve` command).
//!
//! A REST/JSON server over the shared
//! [`controller`](crate::connector::api::controller) layer, so a native app or
//! any HTTP client can drive every memory operation. See [`server::serve`].

mod dream;
mod dream_routes;
mod error;
mod handlers;
mod llm;
mod server;
mod session_import;
mod sessions;

pub use dream::{DreamConfigPatch, DreamService};
pub use server::{routes, serve, AppState};
pub use session_import::SessionImportService;
