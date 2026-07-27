//! Dependency-injection container.
//!
//! Wires the concrete adapters (DuckDB store, session discovery, OpenAI-
//! compatible clients) to the application-layer use cases, resolving the data
//! directory, embedding setup, and LLM endpoint from config / environment once
//! at startup. The router asks the container for fully-built use cases.

use std::sync::{Arc, Mutex};

use openai_rs::{ChatClient, Endpoint};

use crate::application::interfaces::Embedder;
use crate::application::{
    ImportSessionUseCase, MemoryBrowseUseCase, MemoryDreamUseCase, MemoryExtractionUseCase,
    MemoryRepository, MemorySearchUseCase, SessionDiscovery, SummarizeMemoryUseCase,
};
use crate::connector::adapter::{
    build_chat_client, build_embedding_client, DuckdbMemoryRepository, LocalSessionDiscovery,
    MemoryConfig, ResolvedChatEndpoint, ResolvedEmbeddingEndpoint, MEMORY_DB_FILE,
};
use crate::domain::DomainError;

/// Boot-time configuration for the container, assembled from CLI flags.
#[derive(Debug, Clone)]
pub struct ContainerConfig {
    /// Directory holding `memory.duckdb` and `config.json`
    /// (default `~/.memory-rs`).
    pub data_dir: String,
    /// Embedding dimension the database is pinned to on first open.
    pub embedding_dimensions: usize,
    /// Optional `--openai-endpoint` override selecting a named config endpoint.
    pub openai_endpoint: Option<String>,
}

impl ContainerConfig {
    /// Default data directory: `~/.memory-rs`, falling back to `.memory-rs` in
    /// the current directory when no home directory is known.
    pub fn default_data_dir() -> String {
        std::env::var("HOME")
            .map(|h| format!("{h}/.memory-rs"))
            .unwrap_or_else(|_| ".memory-rs".to_string())
    }
}

/// Holds the resolved singletons and hands out use cases.
///
/// The memory repository is opened lazily and cached, so the (potentially
/// migrating) DuckDB open happens once per process and only for commands that
/// touch storage. The embedder is built *alongside* the repository — it embeds
/// queries with the model the **store** was created with (read back from the
/// DB), not merely whatever the current config names, so retrieval always
/// compares vectors from the same model. Its endpoint (base URL / key) still
/// comes from the resolved embedding endpoint, so a remote LLM can pair with
/// local embeddings.
pub struct Container {
    config: ContainerConfig,
    /// Resolved chat endpoint (extraction / summarization / dreaming).
    chat_endpoint: ResolvedChatEndpoint,
    /// Resolved embedding endpoint — its base URL / API key embed queries, and
    /// its model seeds a *fresh* store's pinned embedding model.
    embedding_endpoint: ResolvedEmbeddingEndpoint,
    /// Repository + embedder, opened/built together on first storage access.
    opened: Mutex<Option<Opened>>,
}

/// The lazily-opened storage layer: the repository and the embedder built from
/// the model that store was created with.
#[derive(Clone)]
struct Opened {
    repo: Arc<dyn MemoryRepository>,
    embedder: Embedder,
}

impl Container {
    /// Build the container: load `config.json` and resolve the chat / embedding
    /// endpoints. The repository and embedder are opened lazily on first use.
    pub fn new(config: ContainerConfig) -> Result<Self, DomainError> {
        let file_config = MemoryConfig::load(&config.data_dir)?;
        let embedding_cfg = file_config.embedding();

        // Chat and embeddings resolve independently, so a remote LLM can pair
        // with local embeddings (or vice versa). `--openai-endpoint` overrides
        // both.
        let chat_endpoint = file_config.resolve_chat_endpoint(config.openai_endpoint.as_deref());
        let embedding_endpoint = file_config
            .resolve_embedding_endpoint(config.openai_endpoint.as_deref(), &embedding_cfg.model);

        Ok(Self {
            config,
            chat_endpoint,
            embedding_endpoint,
            opened: Mutex::new(None),
        })
    }

    /// Open (once) the repository and build the embedder from the store's
    /// recorded embedding model. Cached for the process lifetime.
    fn open(&self) -> Result<Opened, DomainError> {
        let mut cache = self
            .opened
            .lock()
            .map_err(|_| DomainError::internal("storage cache lock poisoned"))?;
        if let Some(opened) = cache.as_ref() {
            return Ok(opened.clone());
        }

        let db_path = std::path::Path::new(&self.config.data_dir).join(MEMORY_DB_FILE);
        std::fs::create_dir_all(&self.config.data_dir).map_err(|e| {
            DomainError::storage(format!(
                "failed to create data dir {}: {e}",
                self.config.data_dir
            ))
        })?;

        // The config's embedding model seeds a *fresh* store; an existing store
        // keeps its original. `stored_embedding_model()` returns the effective
        // one, which is what queries must be embedded with.
        let repo = DuckdbMemoryRepository::new(
            &db_path,
            self.config.embedding_dimensions,
            &self.embedding_endpoint.model,
        )?;
        let model = repo.stored_embedding_model().to_string();

        // Embed with the store's model, over the resolved embedding endpoint.
        let client = build_embedding_client(
            &endpoint_from_parts(
                &self.embedding_endpoint.base_url,
                &self.embedding_endpoint.api_key,
            ),
            model,
        )?;
        let opened = Opened {
            repo: Arc::new(repo),
            embedder: Embedder::new(client),
        };
        *cache = Some(opened.clone());
        Ok(opened)
    }

    /// The embedder that embeds queries with the store's recorded model.
    fn embedder(&self) -> Result<Embedder, DomainError> {
        Ok(self.open()?.embedder)
    }

    pub fn data_dir(&self) -> &str {
        &self.config.data_dir
    }

    /// The chat client for the resolved chat endpoint. Errors when that endpoint
    /// names no chat model — the LLM-driven commands (`import`, `dream`, `add`)
    /// need one.
    pub fn chat_client(&self) -> Result<Arc<dyn ChatClient>, DomainError> {
        let ep = &self.chat_endpoint;
        let model = ep.model.clone().ok_or_else(|| {
            DomainError::invalid_input(
                "the chat endpoint has no model; set OPENAI_MODEL or the endpoint's `model`",
            )
        })?;
        Ok(build_chat_client(
            &endpoint_from_parts(&ep.base_url, &ep.api_key),
            model,
        )?)
    }

    /// Open (or return the cached) memory repository. Pins a fresh store to the
    /// configured embedding model + dimensions; an existing store keeps its own.
    pub fn memory_repository(&self) -> Result<Arc<dyn MemoryRepository>, DomainError> {
        Ok(self.open()?.repo)
    }

    /// Session import + memory extraction + virtual-filesystem summarization,
    /// driven by the resolved chat model.
    pub fn memory_import_use_case(&self) -> Result<ImportSessionUseCase, DomainError> {
        let chat_client = self.chat_client()?;
        let memory_repo = self.memory_repository()?;
        let embedder = self.embedder()?;
        let extraction = MemoryExtractionUseCase::new(
            Arc::clone(&chat_client),
            Arc::clone(&memory_repo),
            Arc::new(embedder.clone()),
        );
        let summary =
            SummarizeMemoryUseCase::new(chat_client, Arc::clone(&memory_repo), Arc::new(embedder));
        Ok(ImportSessionUseCase::new(memory_repo, extraction, summary))
    }

    pub fn memory_search_use_case(&self) -> Result<MemorySearchUseCase, DomainError> {
        Ok(MemorySearchUseCase::new(
            self.memory_repository()?,
            Arc::new(self.embedder()?),
        ))
    }

    /// Unified search/browse over items + filesystem nodes (used by the TUI).
    pub fn memory_browse_use_case(&self) -> Result<MemoryBrowseUseCase, DomainError> {
        Ok(MemoryBrowseUseCase::new(
            self.memory_repository()?,
            self.embedder()?,
        ))
    }

    /// Summarization use case (session/resource nodes + digest), driven by the
    /// resolved chat model. Used to add resources and regenerate the digest.
    pub fn memory_summary_use_case(&self) -> Result<SummarizeMemoryUseCase, DomainError> {
        Ok(SummarizeMemoryUseCase::new(
            self.chat_client()?,
            self.memory_repository()?,
            Arc::new(self.embedder()?),
        ))
    }

    /// The dream cycle (harvest finished sessions + consolidate the store),
    /// driven by the resolved chat model.
    pub fn memory_dream_use_case(&self) -> Result<MemoryDreamUseCase, DomainError> {
        let chat_client = self.chat_client()?;
        let memory_repo = self.memory_repository()?;
        let embedder = self.embedder()?;
        let import = self.memory_import_use_case()?;
        let summary = SummarizeMemoryUseCase::new(
            Arc::clone(&chat_client),
            Arc::clone(&memory_repo),
            Arc::new(embedder.clone()),
        );
        Ok(MemoryDreamUseCase::new(
            memory_repo,
            chat_client,
            Arc::new(embedder),
            Arc::new(LocalSessionDiscovery::new(None)),
            import,
            summary,
        ))
    }

    /// Session discovery over the local Claude / OpenCode / Zed stores. Used by
    /// the TUI import screen to list sessions and materialize transcripts.
    pub fn session_discovery(&self) -> Arc<dyn SessionDiscovery> {
        Arc::new(LocalSessionDiscovery::new(None))
    }
}

/// Build an `openai_rs::Endpoint` from a base URL and optional API key.
fn endpoint_from_parts(base_url: &str, api_key: &Option<String>) -> Endpoint {
    Endpoint::new(base_url.to_string()).with_optional_api_key(api_key.clone())
}
