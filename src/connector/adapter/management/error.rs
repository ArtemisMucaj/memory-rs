//! HTTP error type for the management API.
//!
//! Handlers return [`ApiResult`]; a [`DomainError`] converts into an
//! [`ApiError`] with an appropriate status code and a JSON body, so a native
//! app gets structured errors instead of a bare 500.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use crate::domain::DomainError;

/// A management-API error: an HTTP status plus a message.
pub struct ApiError {
    status: StatusCode,
    message: String,
}

pub type ApiResult<T> = Result<T, ApiError>;

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "error": self.message }))).into_response()
    }
}

impl From<DomainError> for ApiError {
    fn from(err: DomainError) -> Self {
        // Map the domain error kind to a sensible HTTP status.
        let status = if err.is_not_found() {
            StatusCode::NOT_FOUND
        } else if err.is_invalid_input() {
            StatusCode::BAD_REQUEST
        } else if err.is_already_exists() {
            StatusCode::CONFLICT
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };
        Self {
            status,
            message: err.to_string(),
        }
    }
}
