//! Command-line interface definitions (clap).
//!
//! Parses flags and subcommands and hands the parsed [`Cli`] to the
//! [`router`](crate::connector::api::router). The LLM/embedding backend is
//! always OpenAI-compatible (via `openai-rs`), resolved from `config.json` or
//! the `OPENAI_*` environment variables — there is no provider flag.

use clap::{Parser, Subcommand, ValueEnum};

use crate::application::DEFAULT_SESSION_LIMIT;
use crate::connector::adapter::DEFAULT_EMBEDDING_DIMENSIONS;

/// clap needs a `const fn`-able default; the constant lives with the use case
/// so the CLI, MCP and HTTP surfaces all brief over the same window.
const fn memory_rs_default_resume_limit() -> usize {
    DEFAULT_SESSION_LIMIT
}

/// Long-term memory for coding assistants: import sessions, ingest durable
/// facts, recall by hybrid semantic+keyword+recency search.
#[derive(Debug, Parser)]
#[command(name = "memory-rs", version, about)]
pub struct Cli {
    /// Data directory holding `memory.duckdb` and `config.json`.
    /// Defaults to `~/.memory-rs`.
    #[arg(long, global = true)]
    pub data_dir: Option<String>,

    /// Embedding dimension the database is pinned to on first open. The
    /// default matches the built-in LM Studio embedding model.
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

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Import a finished session transcript and extract facts from it.
    ///
    /// PATH is a Claude Code session log
    /// (`~/.claude/projects/<project>/<id>.jsonl`) or a generic JSONL chat
    /// log (`{"role": "...", "content": "..."}` per line). Extraction calls
    /// the configured LLM.
    Import {
        /// Path to a transcript file (JSONL).
        path: String,

        /// Re-import even if this session was already imported. This clears
        /// the session's prior memories first — the one destructive
        /// operation in the store.
        #[arg(short, long)]
        force: bool,
    },

    /// Recall stored memories (RRF over semantic, keyword, and recency).
    Search {
        query: String,

        /// Maximum number of results.
        #[arg(long, default_value = "10")]
        num: usize,

        /// Restrict to memories relevant in this project (its memories plus
        /// globals). Omit to search everything.
        #[arg(long)]
        project: Option<String>,

        /// Restrict to a namespace: the union of its member projects'
        /// memories, plus globals. Mutually exclusive with --project.
        #[arg(long, conflicts_with = "project")]
        namespace: Option<String>,

        /// Output format: text or json.
        #[arg(short = 'F', long, value_enum, default_value = "text")]
        format: OutputFormat,
    },

    /// List stored memories, newest first.
    List {
        /// Restrict to one project.
        #[arg(long)]
        project: Option<String>,

        /// Restrict to a namespace. Mutually exclusive with --project.
        #[arg(long, conflicts_with = "project")]
        namespace: Option<String>,

        /// Output format: text or json.
        #[arg(short = 'F', long, value_enum, default_value = "text")]
        format: OutputFormat,
    },

    /// Show one memory by id.
    Show {
        /// A memory ID, or a `memory://resources/<name>` URI.
        id: String,
    },

    /// Forget a memory: hard delete. There is no tombstone — the row is gone.
    Delete {
        /// A memory ID.
        id: String,
    },

    /// List imported sessions.
    Sessions {
        /// Output format: text or json.
        #[arg(short = 'F', long, value_enum, default_value = "text")]
        format: OutputFormat,
    },

    /// Brief yourself on recent work: the latest sessions and what they left
    /// behind, so you can pick up where you stopped without re-explaining.
    Resume {
        /// Restrict to sessions worked in this project.
        #[arg(long)]
        project: Option<String>,

        /// Restrict to a namespace: sessions across its member projects.
        /// Mutually exclusive with --project.
        #[arg(long, conflicts_with = "project")]
        namespace: Option<String>,

        /// How many sessions to cover, newest first.
        #[arg(short, long, default_value_t = memory_rs_default_resume_limit())]
        limit: usize,

        /// Output format: text or json.
        #[arg(short = 'F', long, value_enum, default_value = "text")]
        format: OutputFormat,
    },

    /// Add a resource (a file or a URL) to the memory store.
    ///
    /// Fetches the content, generates a one-line abstract and a longer
    /// overview, and stores it with the full text. Uses the configured LLM
    /// for the summaries.
    Add {
        /// A local file path or an http(s):// URL.
        source: String,

        /// Name (slug) for the resource; derived from the source when
        /// omitted. Reusing a name overwrites that resource.
        #[arg(long)]
        name: Option<String>,
    },

    /// Run one harvest cycle: import finished sessions.
    Dream {
        /// Minutes a session must be inactive to count as finished.
        #[arg(long, default_value = "60")]
        idle_minutes: u64,
    },

    /// List entities known to the store, with their facts.
    Entities {
        /// Output format: text or json.
        #[arg(short = 'F', long, value_enum, default_value = "text")]
        format: OutputFormat,
    },

    /// Show one entity with the facts that anchor to it.
    Entity {
        /// Entity name (or alias).
        name: String,
    },

    /// Manage namespaces — cohesive groups of projects that focus retrieval
    /// across a set of related repositories.
    ///
    /// Namespaces also gate auto-import: a project is harvested only once it
    /// is in one, and only for sessions newer than that namespace.
    Namespace {
        #[command(subcommand)]
        command: NamespaceCommand,
    },

    /// Serve the HTTP management API (and MCP over HTTP) so a native app or
    /// any client can drive memory operations.
    Serve {
        /// Port to listen on.
        #[arg(long, default_value = "8766")]
        port: u16,

        /// Bind 0.0.0.0 (reachable off-host) instead of loopback. The API is
        /// unauthenticated, so this is off by default.
        #[arg(long)]
        public: bool,
    },

    /// Run the MCP server over stdio (for direct assistant integration).
    Mcp,

    /// Launch the interactive terminal UI.
    Tui,
}

#[derive(Debug, Subcommand)]
pub enum NamespaceCommand {
    /// Create an (empty) namespace.
    Create {
        /// Namespace name.
        name: String,
    },

    /// Delete a namespace and its project memberships.
    Delete {
        /// Namespace name.
        name: String,
    },

    /// List namespaces with their project counts.
    List {
        /// Output format: text or json.
        #[arg(short = 'F', long, value_enum, default_value = "text")]
        format: OutputFormat,
    },

    /// Show a namespace's member projects.
    Show {
        /// Namespace name.
        name: String,

        /// Output format: text or json.
        #[arg(short = 'F', long, value_enum, default_value = "text")]
        format: OutputFormat,
    },

    /// Add a project to a namespace.
    Assign {
        /// Namespace name.
        namespace: String,
        /// Project (repository / working-directory name) to add.
        project: String,
    },

    /// Remove a project from a namespace.
    Unassign {
        /// Namespace name.
        namespace: String,
        /// Project to remove.
        project: String,
    },
}
