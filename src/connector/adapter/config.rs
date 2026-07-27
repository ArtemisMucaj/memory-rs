//! On-disk user configuration stored at `<data_dir>/config.json`
//! (default `~/.memory-rs/config.json`).
//!
//! Holds the OpenAI-compatible chat and embedding backends memory-rs talks to
//! (named endpoints plus which one is active), the embedding model + dimension
//! the memory database is pinned to, and the dream-cycle settings. Every
//! section is optional so a partially-written or hand-edited file round-trips
//! cleanly and adding a new section never invalidates an existing file.
//!
//! Endpoint resolution is memory-rs's responsibility — `openai-rs` is a pure
//! client library with no config or environment awareness. When no named
//! endpoint is configured, callers fall back to the `OPENAI_*` environment
//! variables (see [`MemoryConfig::resolve_openai_endpoint`]).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::domain::DomainError;

/// File name (under the resolved data directory) holding user configuration.
const CONFIG_FILE: &str = "config.json";

/// Base URL of the built-in default endpoint — a local LM Studio server. Used
/// when neither a named endpoint nor `OPENAI_BASE_URL` is configured, so the
/// LLM-driven commands work out of the box against a local model.
pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:1234";

/// Default chat model for the built-in endpoint (a local LM Studio model).
pub const DEFAULT_CHAT_MODEL: &str = "google/gemma-4-e2b";

/// Default embedding model when none is configured — the LM Studio
/// `nomic-embed-text-v1.5` model (768-dimensional).
pub const DEFAULT_EMBEDDING_MODEL: &str = "text-embedding-nomic-embed-text-v1.5";

/// Default embedding dimension (output width of [`DEFAULT_EMBEDDING_MODEL`]).
pub const DEFAULT_EMBEDDING_DIMENSIONS: usize = 768;

/// Root configuration document persisted to `<data_dir>/config.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Named OpenAI-compatible endpoints (LM Studio, vLLM, hosted OpenAI, …)
    /// plus which one is active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai: Option<OpenAiConfig>,

    /// Embedding model + dimension the memory database is pinned to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<EmbeddingConfig>,

    /// Dream-cycle settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dream: Option<DreamConfig>,
}

/// A set of named OpenAI-compatible endpoints plus which one is active.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenAiConfig {
    /// Name of the endpoint used when no explicit `--openai-endpoint` is given.
    /// When unset (or naming a missing endpoint), callers fall back to the
    /// `OPENAI_*` environment variables.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active: Option<String>,

    /// Registered endpoints, keyed by a user-chosen name (e.g. `"lmstudio"`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub endpoints: BTreeMap<String, OpenAiEndpoint>,
}

/// One OpenAI-compatible server.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenAiEndpoint {
    /// Base URL, e.g. `http://localhost:1234` (no `/v1` suffix).
    pub base_url: String,

    /// Chat model id sent in chat requests. When absent, the embedding-only
    /// flows still work; chat flows require a model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Embedding model id. When absent, [`DEFAULT_EMBEDDING_MODEL`] is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,

    /// Bearer API key for hosted servers. Absent/empty for local servers like
    /// LM Studio.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}

/// Embedding model + dimension pinned to the memory database on first open.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model id used to embed items, nodes, and queries.
    pub model: String,
    /// Output dimension of `model`. Persisted on first DB open; a later open
    /// with a different value is rejected since vectors would be incomparable.
    pub dimensions: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_EMBEDDING_MODEL.to_string(),
            dimensions: DEFAULT_EMBEDDING_DIMENSIONS,
        }
    }
}

/// Dream-cycle settings. Every field is optional; the accessors apply defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DreamConfig {
    /// Minutes a session must be inactive before it counts as finished and is
    /// harvested (default 60).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_idle_minutes: Option<u64>,
}

impl DreamConfig {
    pub const DEFAULT_SESSION_IDLE_MINUTES: u64 = 60;

    pub fn session_idle_minutes(&self) -> u64 {
        self.session_idle_minutes
            .filter(|m| *m > 0)
            .unwrap_or(Self::DEFAULT_SESSION_IDLE_MINUTES)
    }
}

/// A fully-resolved endpoint ready to build a client from: base URL, optional
/// API key, and the chat/embedding model names. Produced by
/// [`MemoryConfig::resolve_openai_endpoint`] from either a named config
/// endpoint or the `OPENAI_*` environment variables.
#[derive(Debug, Clone)]
pub struct ResolvedEndpoint {
    pub base_url: String,
    pub api_key: Option<String>,
    pub chat_model: Option<String>,
    pub embedding_model: String,
}

impl MemoryConfig {
    /// Absolute path of the config file inside `data_dir`.
    pub fn path_in(data_dir: &str) -> PathBuf {
        Path::new(data_dir).join(CONFIG_FILE)
    }

    /// Load configuration from `<data_dir>/config.json`.
    ///
    /// A missing file yields [`MemoryConfig::default`] so first-run flows work
    /// without a pre-existing file. A present-but-malformed file is an error,
    /// so a user's saved settings are never silently discarded.
    pub fn load(data_dir: &str) -> Result<Self, DomainError> {
        let path = Self::path_in(data_dir);
        let contents = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => {
                return Err(DomainError::internal(format!(
                    "failed to read {}: {e}",
                    path.display()
                )))
            }
        };
        serde_json::from_str(&contents)
            .map_err(|e| DomainError::internal(format!("failed to parse {}: {e}", path.display())))
    }

    /// Persist configuration to `<data_dir>/config.json`, creating the directory
    /// if needed. Written pretty-printed so users can hand-edit it, and made
    /// owner-only since it can hold an API key.
    pub fn save(&self, data_dir: &str) -> Result<(), DomainError> {
        std::fs::create_dir_all(data_dir).map_err(|e| {
            DomainError::internal(format!("failed to create data dir {data_dir}: {e}"))
        })?;
        let path = Self::path_in(data_dir);
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| DomainError::internal(format!("failed to serialize config: {e}")))?;
        std::fs::write(&path, json).map_err(|e| {
            DomainError::internal(format!("failed to write {}: {e}", path.display()))
        })?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).map_err(
                |e| {
                    DomainError::internal(format!(
                        "failed to restrict permissions on {}: {e}",
                        path.display()
                    ))
                },
            )?;
        }
        Ok(())
    }

    /// Mutable access to the OpenAI section, creating it if absent.
    pub fn openai_mut(&mut self) -> &mut OpenAiConfig {
        self.openai.get_or_insert_with(OpenAiConfig::default)
    }

    /// The pinned embedding config (model + dimensions), or the default.
    pub fn embedding(&self) -> EmbeddingConfig {
        self.embedding.clone().unwrap_or_default()
    }

    /// Resolve which endpoint to talk to, honoring `name_override` first, then
    /// the configured `active` endpoint, then the `OPENAI_*` environment
    /// variables, and finally a built-in **local LM Studio** default so the
    /// LLM-driven commands work out of the box with no configuration.
    ///
    /// Because of the built-in default this always resolves to *some* endpoint;
    /// a command still fails clearly at call time if that endpoint is not
    /// actually reachable (e.g. LM Studio isn't running).
    ///
    /// `default_embedding_model` (from config or the built-in default) fills in
    /// the embedding model when an endpoint does not name its own.
    pub fn resolve_openai_endpoint(
        &self,
        name_override: Option<&str>,
        default_embedding_model: &str,
    ) -> Option<ResolvedEndpoint> {
        if let Some(openai) = self.openai.as_ref() {
            if let Some(name) = name_override.or(openai.active.as_deref()) {
                if let Some(ep) = openai.endpoints.get(name) {
                    return Some(ResolvedEndpoint {
                        base_url: ep.base_url.clone(),
                        api_key: ep.api_key.clone(),
                        chat_model: ep.model.clone(),
                        embedding_model: ep
                            .embedding_model
                            .clone()
                            .unwrap_or_else(|| default_embedding_model.to_string()),
                    });
                }
            }
        }
        // Environment overrides, else the built-in local LM Studio default. Each
        // field falls back independently, so setting only `OPENAI_MODEL` (say)
        // still points at the default LM Studio server.
        Some(ResolvedEndpoint {
            base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| DEFAULT_BASE_URL.to_string()),
            api_key: std::env::var("OPENAI_API_KEY").ok(),
            chat_model: Some(
                std::env::var("OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_CHAT_MODEL.to_string()),
            ),
            embedding_model: std::env::var("OPENAI_EMBEDDING_MODEL")
                .unwrap_or_else(|_| default_embedding_model.to_string()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn load_missing_file_returns_default() {
        let dir = TempDir::new().unwrap();
        let cfg = MemoryConfig::load(dir.path().to_str().unwrap()).unwrap();
        assert!(cfg.openai.is_none());
        assert_eq!(cfg.embedding().model, DEFAULT_EMBEDDING_MODEL);
        assert_eq!(cfg.embedding().dimensions, DEFAULT_EMBEDDING_DIMENSIONS);
    }

    #[test]
    fn save_then_load_round_trips() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().to_str().unwrap();

        let mut cfg = MemoryConfig::default();
        cfg.openai_mut().endpoints.insert(
            "lmstudio".to_string(),
            OpenAiEndpoint {
                base_url: "http://localhost:1234".to_string(),
                model: Some("gemma".to_string()),
                embedding_model: Some("nomic-embed".to_string()),
                api_key: None,
            },
        );
        cfg.openai_mut().active = Some("lmstudio".to_string());
        cfg.embedding = Some(EmbeddingConfig {
            model: "nomic-embed".to_string(),
            dimensions: 768,
        });
        cfg.save(data_dir).unwrap();

        let loaded = MemoryConfig::load(data_dir).unwrap();
        assert_eq!(loaded.embedding().dimensions, 768);
        let openai = loaded.openai.as_ref().expect("openai section present");
        assert_eq!(openai.active.as_deref(), Some("lmstudio"));
        let ep = openai.endpoints.get("lmstudio").expect("endpoint present");
        assert_eq!(ep.base_url, "http://localhost:1234");
        assert_eq!(ep.embedding_model.as_deref(), Some("nomic-embed"));
    }

    #[test]
    fn malformed_file_is_an_error() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().to_str().unwrap();
        std::fs::write(MemoryConfig::path_in(data_dir), "{ not json").unwrap();
        assert!(MemoryConfig::load(data_dir).is_err());
    }

    #[test]
    fn empty_config_omits_all_sections() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().to_str().unwrap();
        MemoryConfig::default().save(data_dir).unwrap();
        let written = std::fs::read_to_string(MemoryConfig::path_in(data_dir)).unwrap();
        assert_eq!(written.trim(), "{}", "got: {written}");
    }

    #[test]
    fn resolve_endpoint_precedence_named() {
        let mut cfg = MemoryConfig::default();
        let openai = cfg.openai_mut();
        openai.endpoints.insert(
            "a".to_string(),
            OpenAiEndpoint {
                base_url: "http://a".to_string(),
                ..Default::default()
            },
        );
        openai.endpoints.insert(
            "b".to_string(),
            OpenAiEndpoint {
                base_url: "http://b".to_string(),
                ..Default::default()
            },
        );
        openai.active = Some("a".to_string());

        // Explicit override wins over active.
        assert_eq!(
            cfg.resolve_openai_endpoint(Some("b"), DEFAULT_EMBEDDING_MODEL)
                .unwrap()
                .base_url,
            "http://b"
        );
        // Falls back to active when no override.
        assert_eq!(
            cfg.resolve_openai_endpoint(None, DEFAULT_EMBEDDING_MODEL)
                .unwrap()
                .base_url,
            "http://a"
        );
    }

    #[test]
    fn resolve_endpoint_defaults_embedding_model() {
        let mut cfg = MemoryConfig::default();
        cfg.openai_mut().endpoints.insert(
            "a".to_string(),
            OpenAiEndpoint {
                base_url: "http://a".to_string(),
                ..Default::default()
            },
        );
        cfg.openai_mut().active = Some("a".to_string());
        let resolved = cfg
            .resolve_openai_endpoint(None, "custom-embed")
            .expect("resolves");
        assert_eq!(resolved.embedding_model, "custom-embed");
    }

    #[test]
    fn empty_config_resolves_to_the_builtin_lm_studio_default() {
        // With no named endpoints and no OPENAI_* environment, resolution must
        // fall back to the built-in local LM Studio default so the LLM-driven
        // commands work out of the box. Snapshot/restore the env vars so this
        // test is independent of the caller's shell.
        let saved: Vec<(&str, Option<String>)> = [
            "OPENAI_BASE_URL",
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
            "OPENAI_EMBEDDING_MODEL",
        ]
        .iter()
        .map(|k| (*k, std::env::var(k).ok()))
        .collect();
        for (k, _) in &saved {
            std::env::remove_var(k);
        }

        let cfg = MemoryConfig::default();
        let resolved = cfg
            .resolve_openai_endpoint(None, DEFAULT_EMBEDDING_MODEL)
            .expect("built-in default resolves");

        for (k, v) in saved {
            match v {
                Some(v) => std::env::set_var(k, v),
                None => std::env::remove_var(k),
            }
        }

        assert_eq!(resolved.base_url, DEFAULT_BASE_URL);
        assert_eq!(resolved.chat_model.as_deref(), Some(DEFAULT_CHAT_MODEL));
        assert_eq!(resolved.embedding_model, DEFAULT_EMBEDDING_MODEL);
        assert!(resolved.api_key.is_none());
    }
}
