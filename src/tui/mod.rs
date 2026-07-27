//! Terminal UI (the `memory-rs tui` command).
//!
//! A focused two-screen app — a **Memory** browser and an **Import** picker —
//! built on `ratatui`. A slim top tab bar switches between them. See
//! [`app::App`] for the shell and [`screens`] for each screen.

mod app;
mod markdown;
mod screens;
mod theme;

pub use app::run;
