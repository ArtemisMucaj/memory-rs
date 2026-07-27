//! Thin embedding facade the memory use cases depend on.
//!
//! Memory only ever embeds *text* (item content, node summaries, queries), so
//! it does not need the full [`openai_rs::EmbeddingClient`] surface. This facade
//! wraps an optional client and exposes exactly the two operations the use cases
//! call: [`Embedder::embed_query`] and [`Embedder::embeddings_enabled`].
//!
//! Wrapping the client in an `Option` is what lets memory run in a
//! no-embeddings mode: with no client, items and nodes are still written and
//! stay keyword-searchable, they just carry no vector. This preserves the
//! `Ready` / `Disabled` / `Failed` distinction the write paths rely on (see
//! [`crate::application::use_cases::memory_support`]) without threading an
//! `Option` through every call site.

use std::sync::Arc;

use openai_rs::EmbeddingClient;

use crate::domain::DomainError;

/// Embeds text for semantic recall, or reports that embeddings are disabled.
#[derive(Clone)]
pub struct Embedder {
    client: Option<Arc<dyn EmbeddingClient>>,
}

impl Embedder {
    /// An embedder backed by a live client.
    pub fn new(client: Arc<dyn EmbeddingClient>) -> Self {
        Self {
            client: Some(client),
        }
    }

    /// An embedder that produces no vectors — the no-embeddings mode. Items and
    /// nodes remain keyword-searchable.
    pub fn disabled() -> Self {
        Self { client: None }
    }

    /// Build from an optional client: `Some` ⇒ enabled, `None` ⇒ disabled.
    pub fn from_optional(client: Option<Arc<dyn EmbeddingClient>>) -> Self {
        Self { client }
    }

    /// `false` when no client is configured, so callers skip the embed stage and
    /// write vectors of `None` instead of calling [`Self::embed_query`] (which
    /// would error).
    pub fn embeddings_enabled(&self) -> bool {
        self.client.is_some()
    }

    /// Embed a single piece of text into a query vector.
    ///
    /// Errors when embeddings are disabled — callers must gate on
    /// [`Self::embeddings_enabled`] first, matching the original
    /// `EmbeddingService` contract.
    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>, DomainError> {
        match &self.client {
            Some(client) => Ok(client.embed(text).await?),
            None => Err(DomainError::embedding(
                "embeddings are disabled (no embedding client configured)",
            )),
        }
    }
}
