//! Long-term memory for coding assistants.
//!
//! Imports finished assistant-session transcripts, extracts durable memories
//! (preferences / experiences / skills / facts), builds a `memory://` virtual
//! filesystem of L0/L1/L2 nodes over them, runs a "dream" consolidation cycle,
//! and answers hybrid semantic+keyword recall — all stored in its own
//! `memory.duckdb`.
//!
//! # Layering
//!
//! Dependencies point inward, following ports & adapters:
//!
//! - [`domain`] — pure value types ([`MemoryItem`], [`MemoryNode`],
//!   [`SessionTranscript`], [`Memory`], [`DomainError`], …). No I/O, no async.
//! - [`application`] — use cases (orchestration) and port traits
//!   ([`NodeRepository`], [`SessionDiscovery`]). Depends only on the domain.
//! - [`connector`] — concrete adapters (DuckDB store, session discovery,
//!   transcript parsing, resource fetch). Depends on application + domain.
//!
//! The LLM and embedding backends come from the [`openai_rs`] crate: use cases
//! consume its [`ChatClient`](openai_rs::ChatClient) and
//! [`EmbeddingClient`](openai_rs::EmbeddingClient) ports directly, and the
//! connector layer builds its OpenAI-compatible adapters.

pub mod application;
pub mod cli;
pub mod connector;
pub mod domain;
pub mod tui;

pub use application::{
    resource_slug, DreamReport, ExtractionReport, HarvestReport, ImportOutcome,
    ImportSessionUseCase, IngestionOutcome, IngestionReport, MemoryBrowseUseCase,
    MemoryDreamUseCase, MemoryExtractionUseCase, MemoryIngestionUseCase, MemoryLevel,
    MemoryRecallUseCase, MemoryRepository, MemoryRow, MemorySearchUseCase, NodeRepository,
    NodeStats, RowTarget, SessionDiscovery, SummarizeMemoryUseCase, MEMORY_ROOT_URI,
    PROJECTS_ROOT_URI, RESOURCES_ROOT_URI, SESSIONS_ROOT_URI,
};

pub use connector::{
    build_chat_client, build_embedding_client, fetch_resource, parse_transcript,
    parse_transcript_file, DiscoveredSessionSources, DuckdbStore, FetchedResource, MEMORY_DB_FILE,
};

pub use domain::{
    cosine_similarity, DiscoveredSession, DomainError, DreamRun, EdgeOrigin, EdgeType, Entity,
    EntityRef, ImportedSession, Memory, MemoryEdge, MemoryItem, MemoryKind, MemoryNode,
    MemoryOperation, MemoryStatus, MemoryStoreStats, NodeKind, Predicate, SessionLocator,
    SessionMessage, SessionSource, SessionStatus, SessionTranscript, SourceKind,
};
