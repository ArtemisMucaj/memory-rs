//! Long-term memory for coding assistants.
//!
//! Imports finished assistant-session transcripts, extracts durable facts
//! (subject–predicate–object triples anchored to resolved entities), and
//! answers hybrid semantic+keyword+recency recall — all stored in its own
//! `memory.duckdb`.
//!
//! # Layering
//!
//! Dependencies point inward, following ports & adapters:
//!
//! - [`domain`] — pure value types ([`Memory`], [`Entity`], [`Predicate`],
//!   [`SessionTranscript`], [`DomainError`], …). No I/O, no async.
//! - [`application`] — use cases (orchestration) and port traits
//!   ([`MemoryRepository`], [`SessionDiscovery`]). Depends only on the domain.
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
    DreamReport, HarvestReport, ImportOutcome, ImportSessionUseCase, IngestionOutcome,
    IngestionReport, MemoryDreamUseCase, MemoryIngestionUseCase, MemoryRecallUseCase,
    MemoryRepository, MemoryResumeUseCase, Recalled, ResumeBriefing, SessionDiscovery,
    SessionRecap, DEFAULT_SESSION_LIMIT, MAX_SESSION_LIMIT,
};

pub use connector::{
    build_chat_client, build_embedding_client, fetch_resource, parse_transcript,
    parse_transcript_file, DiscoveredSessionSources, DuckdbStore, FetchedResource, MEMORY_DB_FILE,
};

pub use domain::{
    cosine_similarity, entity_name_key, DiscoveredSession, DomainError, Entity, ImportedSession,
    Memory, MemoryKind, MemoryResource, SessionLocator, SessionMessage, SessionSource,
    SessionStatus, SessionTranscript, SourceKind, VALID_ENTITY_TYPES,
};
