use thiserror::Error;

#[derive(Debug, Error)]
pub enum DomainError {
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Already exists: {0}")]
    AlreadyExists(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl DomainError {
    pub fn storage(msg: impl Into<String>) -> Self {
        Self::StorageError(msg.into())
    }

    pub fn parse(msg: impl Into<String>) -> Self {
        Self::ParseError(msg.into())
    }

    pub fn embedding(msg: impl Into<String>) -> Self {
        Self::EmbeddingError(msg.into())
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }

    pub fn already_exists(msg: impl Into<String>) -> Self {
        Self::AlreadyExists(msg.into())
    }

    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }

    pub fn is_not_found(&self) -> bool {
        matches!(self, Self::NotFound(_))
    }

    pub fn is_invalid_input(&self) -> bool {
        matches!(self, Self::InvalidInput(_))
    }

    pub fn is_already_exists(&self) -> bool {
        matches!(self, Self::AlreadyExists(_))
    }

    pub fn is_storage_error(&self) -> bool {
        matches!(self, Self::StorageError(_))
    }
}

impl From<openai_rs::OpenAiError> for DomainError {
    /// Bridge the LLM/embedding transport errors into the memory domain's error
    /// space. A model or embedding call that fails surfaces as an
    /// [`DomainError::EmbeddingError`] at the boundary — the use cases that call
    /// through `openai_rs` ports never see the transport type.
    fn from(err: openai_rs::OpenAiError) -> Self {
        Self::EmbeddingError(err.to_string())
    }
}
