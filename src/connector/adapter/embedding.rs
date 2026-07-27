//! Builders for OpenAI-compatible chat and embedding clients.

use std::sync::Arc;

use openai_rs::{ChatClient, EmbeddingClient, Endpoint, OpenAiError};

/// Build an `OpenAiChatClient` for `model` from an endpoint configuration.
pub fn build_chat_client(
    endpoint: &Endpoint,
    model: impl Into<String>,
) -> Result<Arc<dyn ChatClient>, OpenAiError> {
    let client = openai_rs::OpenAiChatClient::new(endpoint, model)?;
    Ok(Arc::new(client))
}

/// Build an `OpenAiEmbeddingClient` for `model` from an endpoint configuration.
pub fn build_embedding_client(
    endpoint: &Endpoint,
    model: impl Into<String>,
) -> Result<Arc<dyn EmbeddingClient>, OpenAiError> {
    let client = openai_rs::OpenAiEmbeddingClient::new(endpoint, model)?;
    Ok(Arc::new(client))
}
