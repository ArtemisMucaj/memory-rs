//! GitHub Copilot as a chat backend.
//!
//! The Copilot API is OpenAI-compatible but served at the root (no `/v1`) behind
//! a set of client-identity headers. That Copilot-specific knowledge lives in
//! [`gh_copilot_rs`]; this module wires it to openai-rs's chat client so the
//! shared chat/stream logic is reused rather than duplicated, and exposes the
//! model catalog for the picker.
//!
//! Auth is the GitHub OAuth **device flow**, run over the management API (see
//! `management::copilot_login`); the captured `ghu_…` token lives in
//! `config.json`'s `copilot` section.

use std::sync::Arc;

use gh_copilot_rs::{CopilotEndpoint, CopilotModel, CopilotModelCatalog, CopilotToken};
use openai_rs::{ApiRoutes, ChatClient, Endpoint, OpenAiChatClient, Transport};

use crate::connector::adapter::CopilotConfig;
use crate::domain::DomainError;

/// Build a [`ChatClient`] that routes completions through a Copilot
/// subscription.
///
/// The token is required: without it every request 401s, which surfaces as an
/// opaque failure deep inside an import. Fail here with something actionable
/// instead.
pub fn chat_client(config: &CopilotConfig) -> Result<Arc<dyn ChatClient>, DomainError> {
    if !config.is_authenticated() {
        return Err(DomainError::invalid_input(
            "GitHub Copilot is selected for chat but not authenticated — run the Copilot login first",
        ));
    }

    let endpoint = copilot_endpoint(config);
    // Wire the Copilot endpoint to an OpenAI-compatible client: root-served
    // routes, the Copilot protocol headers, and the token as the bearer key.
    let openai_endpoint = Endpoint::new(endpoint.base_url())
        .with_routes(ApiRoutes::unversioned())
        .with_headers(endpoint.protocol_headers())
        .with_timeout(endpoint.timeout())
        .with_optional_api_key(endpoint.token().map(|t| t.expose().to_string()));

    let transport = Transport::new(&openai_endpoint)
        .map_err(|e| DomainError::internal(format!("failed to build the Copilot transport: {e}")))?;
    let model = config.model.clone().unwrap_or_default();
    Ok(Arc::new(OpenAiChatClient::with_transport(transport, model)))
}

/// The Copilot endpoint description for the configured token.
fn copilot_endpoint(config: &CopilotConfig) -> CopilotEndpoint {
    CopilotEndpoint::from_optional_token(config.github_token.clone().map(CopilotToken::new))
}

/// Every model the subscription offers, for the model picker.
pub async fn list_models(config: &CopilotConfig) -> Result<Vec<CopilotModel>, DomainError> {
    if !config.is_authenticated() {
        return Err(DomainError::invalid_input(
            "not authenticated with GitHub Copilot — run the Copilot login first",
        ));
    }
    let endpoint = copilot_endpoint(config);
    let catalog = CopilotModelCatalog::from_endpoint(&endpoint).map_err(|e| {
        DomainError::internal(format!("failed to build the Copilot model catalog: {e}"))
    })?;
    catalog
        .list_models()
        .await
        .map_err(|e| DomainError::invalid_input(format!("could not list Copilot models: {e}")))
}
