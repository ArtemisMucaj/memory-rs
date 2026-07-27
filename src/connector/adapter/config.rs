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
//! client library with no config or environment awareness. Chat and embeddings
//! resolve independently (see [`MemoryConfig::resolve_chat_endpoint`] and
//! [`MemoryConfig::resolve_embedding_endpoint`]), so a remote LLM can be paired
//! with local embeddings; each falls back to the `OPENAI_*` environment
//! variables, then a built-in local LM Studio default.

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

/// A set of named OpenAI-compatible endpoints plus which ones are active.
///
/// Chat and embeddings resolve **independently**, so a remote LLM can be paired
/// with local embeddings (or vice versa): [`active_chat`](Self::active_chat) and
/// [`active_embedding`](Self::active_embedding) each name a registered endpoint.
/// When one is unset it falls back to [`active`](Self::active) (the shared
/// default), so a single `active` still drives both — the common case.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenAiConfig {
    /// Shared default endpoint used for whichever of chat / embeddings does not
    /// name its own. When unset (or naming a missing endpoint), callers fall
    /// back to the `OPENAI_*` environment variables, then the built-in default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active: Option<String>,

    /// Endpoint used for chat (extraction / summarization / dreaming).
    /// Falls back to [`active`](Self::active) when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_chat: Option<String>,

    /// Endpoint used for embeddings (semantic recall). Falls back to
    /// [`active`](Self::active) when unset. Point this at a local server to run
    /// embeddings locally while chat goes to a remote model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_embedding: Option<String>,

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

/// A fully-resolved chat endpoint: where to send chat requests and with which
/// model. Produced by [`MemoryConfig::resolve_chat_endpoint`].
#[derive(Debug, Clone)]
pub struct ResolvedChatEndpoint {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: Option<String>,
}

/// A fully-resolved embedding endpoint: where to send embedding requests and
/// with which model. Produced by [`MemoryConfig::resolve_embedding_endpoint`].
#[derive(Debug, Clone)]
pub struct ResolvedEmbeddingEndpoint {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
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

    /// Select the named endpoint for a role: `name_override` first, then the
    /// role-specific active (`active_chat` / `active_embedding`), then the
    /// shared `active`. Returns the endpoint config, or `None` to fall through
    /// to the environment / built-in default.
    fn select_endpoint(
        &self,
        name_override: Option<&str>,
        role_active: fn(&OpenAiConfig) -> Option<&str>,
    ) -> Option<&OpenAiEndpoint> {
        let openai = self.openai.as_ref()?;
        let name = name_override
            .or_else(|| role_active(openai))
            .or(openai.active.as_deref())?;
        openai.endpoints.get(name)
    }

    /// Resolve the **chat** endpoint: the named config endpoint (override →
    /// `active_chat` → `active`), else the `OPENAI_*` environment variables,
    /// else the built-in **local LM Studio** default. Always resolves to some
    /// endpoint; a command still fails clearly at call time if it is not
    /// reachable (e.g. LM Studio isn't running).
    pub fn resolve_chat_endpoint(&self, name_override: Option<&str>) -> ResolvedChatEndpoint {
        if let Some(ep) = self.select_endpoint(name_override, |o| o.active_chat.as_deref()) {
            return ResolvedChatEndpoint {
                base_url: ep.base_url.clone(),
                api_key: ep.api_key.clone(),
                model: ep.model.clone(),
            };
        }
        ResolvedChatEndpoint {
            base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| DEFAULT_BASE_URL.to_string()),
            api_key: std::env::var("OPENAI_API_KEY").ok(),
            model: Some(
                std::env::var("OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_CHAT_MODEL.to_string()),
            ),
        }
    }

    /// Resolve the **embedding** endpoint independently of chat: the named
    /// config endpoint (override → `active_embedding` → `active`), else the
    /// `OPENAI_EMBEDDING_*` (then `OPENAI_*`) environment variables, else the
    /// built-in local LM Studio default. Point `active_embedding` at a local
    /// server to run embeddings locally while chat goes to a remote model.
    ///
    /// `default_embedding_model` (from config or the built-in default) fills in
    /// the model when the endpoint does not name its own `embedding_model`.
    pub fn resolve_embedding_endpoint(
        &self,
        name_override: Option<&str>,
        default_embedding_model: &str,
    ) -> ResolvedEmbeddingEndpoint {
        if let Some(ep) = self.select_endpoint(name_override, |o| o.active_embedding.as_deref()) {
            return ResolvedEmbeddingEndpoint {
                base_url: ep.base_url.clone(),
                api_key: ep.api_key.clone(),
                model: ep
                    .embedding_model
                    .clone()
                    .unwrap_or_else(|| default_embedding_model.to_string()),
            };
        }
        // A dedicated OPENAI_EMBEDDING_BASE_URL / _API_KEY lets embeddings target
        // a different server than chat purely from the environment; each falls
        // back to the chat-shared OPENAI_* var, then the built-in default.
        ResolvedEmbeddingEndpoint {
            base_url: std::env::var("OPENAI_EMBEDDING_BASE_URL")
                .or_else(|_| std::env::var("OPENAI_BASE_URL"))
                .unwrap_or_else(|_| DEFAULT_BASE_URL.to_string()),
            api_key: std::env::var("OPENAI_EMBEDDING_API_KEY")
                .or_else(|_| std::env::var("OPENAI_API_KEY"))
                .ok(),
            model: std::env::var("OPENAI_EMBEDDING_MODEL")
                .unwrap_or_else(|_| default_embedding_model.to_string()),
        }
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

    fn endpoint(base_url: &str) -> OpenAiEndpoint {
        OpenAiEndpoint {
            base_url: base_url.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn resolve_endpoint_precedence_named() {
        let mut cfg = MemoryConfig::default();
        let openai = cfg.openai_mut();
        openai
            .endpoints
            .insert("a".to_string(), endpoint("http://a"));
        openai
            .endpoints
            .insert("b".to_string(), endpoint("http://b"));
        openai.active = Some("a".to_string());

        // Explicit override wins over active (chat and embedding alike).
        assert_eq!(cfg.resolve_chat_endpoint(Some("b")).base_url, "http://b");
        assert_eq!(
            cfg.resolve_embedding_endpoint(Some("b"), DEFAULT_EMBEDDING_MODEL)
                .base_url,
            "http://b"
        );
        // Falls back to the shared `active` when no override or role-specific.
        assert_eq!(cfg.resolve_chat_endpoint(None).base_url, "http://a");
    }

    #[test]
    fn chat_and_embedding_can_target_different_servers() {
        // The headline use case: a remote LLM for chat, local embeddings.
        let mut cfg = MemoryConfig::default();
        let openai = cfg.openai_mut();
        openai
            .endpoints
            .insert("remote".to_string(), endpoint("https://api.example.com"));
        openai
            .endpoints
            .insert("local".to_string(), endpoint("http://127.0.0.1:1234"));
        openai.active_chat = Some("remote".to_string());
        openai.active_embedding = Some("local".to_string());

        assert_eq!(
            cfg.resolve_chat_endpoint(None).base_url,
            "https://api.example.com"
        );
        assert_eq!(
            cfg.resolve_embedding_endpoint(None, DEFAULT_EMBEDDING_MODEL)
                .base_url,
            "http://127.0.0.1:1234"
        );
    }

    #[test]
    fn role_active_falls_back_to_shared_active() {
        // Only `active_chat` is set; embeddings fall through to `active`.
        let mut cfg = MemoryConfig::default();
        let openai = cfg.openai_mut();
        openai
            .endpoints
            .insert("remote".to_string(), endpoint("https://api.example.com"));
        openai
            .endpoints
            .insert("shared".to_string(), endpoint("http://shared"));
        openai.active = Some("shared".to_string());
        openai.active_chat = Some("remote".to_string());

        assert_eq!(
            cfg.resolve_chat_endpoint(None).base_url,
            "https://api.example.com"
        );
        assert_eq!(
            cfg.resolve_embedding_endpoint(None, DEFAULT_EMBEDDING_MODEL)
                .base_url,
            "http://shared"
        );
    }

    #[test]
    fn resolve_endpoint_defaults_embedding_model() {
        let mut cfg = MemoryConfig::default();
        cfg.openai_mut()
            .endpoints
            .insert("a".to_string(), endpoint("http://a"));
        cfg.openai_mut().active = Some("a".to_string());
        let resolved = cfg.resolve_embedding_endpoint(None, "custom-embed");
        assert_eq!(resolved.model, "custom-embed");
    }

    #[test]
    fn empty_config_resolves_to_the_builtin_lm_studio_default() {
        // With no named endpoints and no OPENAI_* environment, both chat and
        // embeddings fall back to the built-in local LM Studio default so the
        // LLM-driven commands work out of the box. Snapshot/restore the env vars
        // so this test is independent of the caller's shell.
        let keys = [
            "OPENAI_BASE_URL",
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
            "OPENAI_EMBEDDING_MODEL",
            "OPENAI_EMBEDDING_BASE_URL",
            "OPENAI_EMBEDDING_API_KEY",
        ];
        let saved: Vec<(&str, Option<String>)> =
            keys.iter().map(|k| (*k, std::env::var(k).ok())).collect();
        for k in keys {
            std::env::remove_var(k);
        }

        let cfg = MemoryConfig::default();
        let chat = cfg.resolve_chat_endpoint(None);
        let embed = cfg.resolve_embedding_endpoint(None, DEFAULT_EMBEDDING_MODEL);

        for (k, v) in saved {
            match v {
                Some(v) => std::env::set_var(k, v),
                None => std::env::remove_var(k),
            }
        }

        assert_eq!(chat.base_url, DEFAULT_BASE_URL);
        assert_eq!(chat.model.as_deref(), Some(DEFAULT_CHAT_MODEL));
        assert!(chat.api_key.is_none());
        assert_eq!(embed.base_url, DEFAULT_BASE_URL);
        assert_eq!(embed.model, DEFAULT_EMBEDDING_MODEL);
    }
}
