//! Command-line interface definitions (clap).
//!
//! Parses flags and subcommands and hands the parsed [`Cli`] to the
//! [`router`](crate::connector::api::router). The LLM/embedding backend is
//! always OpenAI-compatible (via `openai-rs`), resolved from `config.json` or
//! the `OPENAI_*` environment variables — there is no provider flag.

use clap::{Parser, Subcommand, ValueEnum};

use crate::connector::adapter::DEFAULT_EMBEDDING_DIMENSIONS;
use crate::domain::{MemoryKind, NodeKind};

/// Long-term memory for coding assistants: import sessions, extract durable
/// memories, recall by hybrid search.
#[derive(Debug, Parser)]
#[command(name = "memory-rs", version, about)]
pub struct Cli {
    /// Data directory holding `memory.duckdb` and `config.json`.
    /// Defaults to `~/.memory-rs`.
    #[arg(long, global = true)]
    pub data_dir: Option<String>,

    /// Embedding dimension the database is pinned to on first open. The default
    /// matches the built-in LM Studio embedding model.
    #[arg(long, global = true, default_value_t = DEFAULT_EMBEDDING_DIMENSIONS)]
    pub embedding_dimensions: usize,

    /// Use a named OpenAI endpoint from `config.json` instead of the active one.
    #[arg(long, global = true)]
    pub openai_endpoint: Option<String>,

    #[command(subcommand)]
    pub command: Command,
}

/// Text or JSON output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
}

/// A memory kind as a CLI value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum MemoryKindArg {
    Preference,
    Experience,
    Skill,
    Fact,
}

impl From<MemoryKindArg> for MemoryKind {
    fn from(k: MemoryKindArg) -> Self {
        match k {
            MemoryKindArg::Preference => MemoryKind::Preference,
            MemoryKindArg::Experience => MemoryKind::Experience,
            MemoryKindArg::Skill => MemoryKind::Skill,
            MemoryKindArg::Fact => MemoryKind::Fact,
        }
    }
}

/// A node kind as a CLI value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum NodeKindArg {
    Memory,
    Project,
    Session,
    Resource,
}

impl From<NodeKindArg> for NodeKind {
    fn from(k: NodeKindArg) -> Self {
        match k {
            NodeKindArg::Memory => NodeKind::Memory,
            NodeKindArg::Project => NodeKind::Project,
            NodeKindArg::Session => NodeKind::Session,
            NodeKindArg::Resource => NodeKind::Resource,
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Import a finished session transcript and extract memories from it.
    ///
    /// PATH is a Claude Code session log
    /// (`~/.claude/projects/<project>/<id>.jsonl`) or a generic JSONL chat log
    /// (`{"role": "...", "content": "..."}` per line). Extraction calls the
    /// configured LLM.
    Import {
        /// Path to a transcript file (JSONL).
        path: String,

        /// Re-import even if this session was already imported.
        #[arg(short, long)]
        force: bool,
    },

    /// Search stored memories (hybrid semantic + keyword).
    Search {
        query: String,

        /// Maximum number of results.
        #[arg(long, default_value = "10")]
        num: usize,

        /// Restrict to one memory kind.
        #[arg(short, long, value_enum)]
        kind: Option<MemoryKindArg>,

        /// Restrict to memories relevant in this project/namespace (its items
        /// plus globals). Omit to search everything.
        #[arg(long)]
        project: Option<String>,

        /// Output format: text or json.
        #[arg(short = 'F', long, value_enum, default_value = "text")]
        format: OutputFormat,
    },

    /// List stored memories, newest first.
    List {
        /// Restrict to one memory kind.
        #[arg(short, long, value_enum)]
        kind: Option<MemoryKindArg>,

        /// Output format: text or json.
        #[arg(short = 'F', long, value_enum, default_value = "text")]
        format: OutputFormat,
    },

    /// Show the full content of one memory item or virtual-filesystem node.
    Show {
        /// Memory item ID, a `kind/name` item reference, or a `memory://` node
        /// URI (e.g. `memory://memory`, `memory://sessions/<id>`).
        id: String,
    },

    /// Delete a memory item by ID, or a `kind/name` reference.
    Delete {
        /// Memory item ID or a `kind/name` reference.
        id: String,
    },

    /// List imported sessions.
    Sessions {
        /// Output format: text or json.
        #[arg(short = 'F', long, value_enum, default_value = "text")]
        format: OutputFormat,
    },

    /// Add a resource (a file or a URL) to the memory virtual filesystem.
    ///
    /// Fetches the content, generates an L0 abstract + L1 overview, and stores
    /// it at `memory://resources/<name>` with the full text as L2. Uses the
    /// configured LLM for the summary.
    Add {
        /// A local file path or an http(s):// URL.
        source: String,

        /// Name (slug) for the resource node; derived from the source when
        /// omitted. Reusing a name overwrites that resource.
        #[arg(long)]
        name: Option<String>,
    },

    /// Run one dream cycle: harvest finished sessions, then consolidate the
    /// memory store.
    Dream {
        /// Minutes a session must be inactive to count as finished.
        #[arg(long, default_value = "60")]
        idle_minutes: u64,
    },

    /// Browse the memory virtual filesystem (L0/L1 abstracts).
    ///
    /// With no URI, lists the top-level roots. With a directory URI, lists its
    /// children with their one-line abstracts.
    Tree {
        /// Directory URI to list (e.g. `memory://sessions`). Omit for the root.
        uri: Option<String>,

        /// Output format: text or json.
        #[arg(short = 'F', long, value_enum, default_value = "text")]
        format: OutputFormat,
    },

    /// Show memory-store statistics.
    Stats {
        /// Output format: text or json.
        #[arg(short = 'F', long, value_enum, default_value = "text")]
        format: OutputFormat,
    },

    /// Launch the interactive terminal UI (Memory browser + Import).
    Tui,
}
