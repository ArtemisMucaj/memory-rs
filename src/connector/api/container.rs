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
    MemoryConfig, ResolvedEndpoint, MEMORY_DB_FILE,
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
/// touch storage. The chat client and embedder are built eagerly at
/// construction from the resolved endpoint.
pub struct Container {
    config: ContainerConfig,
    /// The resolved endpoint, if a backend is configured; `None` runs
    /// embeddings-disabled and errors on commands that need an LLM.
    endpoint: Option<ResolvedEndpoint>,
    embedder: Embedder,
    memory_repo: Mutex<Option<Arc<dyn MemoryRepository>>>,
}

impl Container {
    /// Build the container: load `config.json`, resolve the endpoint, and build
    /// the embedder. A missing/embeddings-disabled setup is not an error here —
    /// keyword-only flows still work; commands that require embeddings or chat
    /// surface a clear error at call time.
    pub fn new(config: ContainerConfig) -> Result<Self, DomainError> {
        let file_config = MemoryConfig::load(&config.data_dir)?;
        let embedding_cfg = file_config.embedding();
        let endpoint = file_config
            .resolve_openai_endpoint(config.openai_endpoint.as_deref(), &embedding_cfg.model);

        let embedder = match &endpoint {
            Some(ep) => {
                let client = build_embedding_client(
                    &endpoint_from_resolved(ep),
                    ep.embedding_model.clone(),
                )?;
                Embedder::new(client)
            }
            None => Embedder::disabled(),
        };

        Ok(Self {
            config,
            endpoint,
            embedder,
            memory_repo: Mutex::new(None),
        })
    }

    pub fn data_dir(&self) -> &str {
        &self.config.data_dir
    }

    /// The chat client for the resolved endpoint. Errors when no backend is
    /// configured or the endpoint names no chat model — the LLM-driven commands
    /// (`import`, `dream`, `add`) need it.
    pub fn chat_client(&self) -> Result<Arc<dyn ChatClient>, DomainError> {
        let ep = self.endpoint.as_ref().ok_or_else(|| {
            DomainError::invalid_input(
                "no LLM backend configured; set OPENAI_BASE_URL or add an endpoint to config.json",
            )
        })?;
        let model = ep.chat_model.clone().ok_or_else(|| {
            DomainError::invalid_input(
                "the active endpoint has no chat model; set OPENAI_MODEL or the endpoint's `model`",
            )
        })?;
        Ok(build_chat_client(&endpoint_from_resolved(ep), model)?)
    }

    /// Open (or return the cached) memory repository, pinned to the configured
    /// embedding model + dimensions.
    pub fn memory_repository(&self) -> Result<Arc<dyn MemoryRepository>, DomainError> {
        let mut cache = self
            .memory_repo
            .lock()
            .map_err(|_| DomainError::internal("memory repository cache lock poisoned"))?;
        if let Some(repo) = cache.as_ref() {
            return Ok(Arc::clone(repo));
        }
        let db_path = std::path::Path::new(&self.config.data_dir).join(MEMORY_DB_FILE);
        std::fs::create_dir_all(&self.config.data_dir).map_err(|e| {
            DomainError::storage(format!(
                "failed to create data dir {}: {e}",
                self.config.data_dir
            ))
        })?;
        let embedding_model = self
            .endpoint
            .as_ref()
            .map(|e| e.embedding_model.clone())
            .unwrap_or_else(|| "none".to_string());
        let repo: Arc<dyn MemoryRepository> = Arc::new(DuckdbMemoryRepository::new(
            &db_path,
            self.config.embedding_dimensions,
            &embedding_model,
        )?);
        *cache = Some(Arc::clone(&repo));
        Ok(repo)
    }

    /// Session import + memory extraction + virtual-filesystem summarization,
    /// driven by the resolved chat model.
    pub fn memory_import_use_case(&self) -> Result<ImportSessionUseCase, DomainError> {
        let chat_client = self.chat_client()?;
        let memory_repo = self.memory_repository()?;
        let extraction = MemoryExtractionUseCase::new(
            Arc::clone(&chat_client),
            Arc::clone(&memory_repo),
            Arc::new(self.embedder.clone()),
        );
        let summary = SummarizeMemoryUseCase::new(
            chat_client,
            Arc::clone(&memory_repo),
            Arc::new(self.embedder.clone()),
        );
        Ok(ImportSessionUseCase::new(memory_repo, extraction, summary))
    }

    pub fn memory_search_use_case(&self) -> Result<MemorySearchUseCase, DomainError> {
        Ok(MemorySearchUseCase::new(
            self.memory_repository()?,
            Arc::new(self.embedder.clone()),
        ))
    }

    /// Unified search/browse over items + filesystem nodes (used by the TUI).
    pub fn memory_browse_use_case(&self) -> Result<MemoryBrowseUseCase, DomainError> {
        Ok(MemoryBrowseUseCase::new(
            self.memory_repository()?,
            self.embedder.clone(),
        ))
    }

    /// Summarization use case (session/resource nodes + digest), driven by the
    /// resolved chat model. Used to add resources and regenerate the digest.
    pub fn memory_summary_use_case(&self) -> Result<SummarizeMemoryUseCase, DomainError> {
        Ok(SummarizeMemoryUseCase::new(
            self.chat_client()?,
            self.memory_repository()?,
            Arc::new(self.embedder.clone()),
        ))
    }

    /// The dream cycle (harvest finished sessions + consolidate the store),
    /// driven by the resolved chat model.
    pub fn memory_dream_use_case(&self) -> Result<MemoryDreamUseCase, DomainError> {
        let chat_client = self.chat_client()?;
        let memory_repo = self.memory_repository()?;
        let import = self.memory_import_use_case()?;
        let summary = SummarizeMemoryUseCase::new(
            Arc::clone(&chat_client),
            Arc::clone(&memory_repo),
            Arc::new(self.embedder.clone()),
        );
        Ok(MemoryDreamUseCase::new(
            memory_repo,
            chat_client,
            Arc::new(self.embedder.clone()),
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

/// Build an `openai_rs::Endpoint` from a resolved config/env endpoint.
fn endpoint_from_resolved(ep: &ResolvedEndpoint) -> Endpoint {
    Endpoint::new(ep.base_url.clone()).with_optional_api_key(ep.api_key.clone())
}
