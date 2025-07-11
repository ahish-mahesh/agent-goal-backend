//! # Error Handling
//! 
//! This module defines custom error types and how they're converted to HTTP responses.
//! This is a great example of Rust's powerful error handling system.
//! 
//! ## Key Rust Concepts for Error Handling:
//! 
//! ### Result<T, E> Type
//! - **Purpose**: Forces you to handle both success and failure cases
//! - **T**: The success type (what you get when everything works)
//! - **E**: The error type (what you get when something goes wrong)
//! - **No exceptions**: Rust doesn't have try/catch, it uses Result instead
//! 
//! ### Enums for Error Types
//! - **Variants**: Each enum variant represents a different kind of error
//! - **Data**: Each variant can hold additional information (String, numbers, etc.)
//! - **Pattern matching**: Use `match` to handle different error types
//! 
//! ### Traits for Error Conversion
//! - **From trait**: Automatically converts between error types
//! - **ResponseError trait**: Converts errors to HTTP responses
//! - **Display trait**: Defines how errors are formatted as strings
//! 
//! ## Why custom errors:
//! Custom error types make it easy to handle different failure scenarios
//! and provide meaningful error messages to API clients.

use actix_web::{HttpResponse, ResponseError};  // Web framework error handling
use serde_json::json;                          // For creating JSON error responses
use std::fmt;                                  // For implementing Display trait

/// Custom error types for the application.
/// 
/// ## Rust Concepts:
/// - **enum**: A type that can be one of several variants
/// - **String**: Each variant holds an error message
/// - **#[derive(Debug)]**: Automatically implements debug printing
/// 
/// ## Error Categories:
/// - **Internal**: Server-side problems (500 errors)
/// - **BadRequest**: Client sent invalid data (400 errors)
/// - **NotFound**: Requested resource doesn't exist (404 errors)
/// - **ConfigError**: Configuration problems (500 errors)
/// - **ValidationError**: Data validation failed (400 errors)
/// 
/// ## Usage Example:
/// ```rust
/// return Err(AppError::BadRequest("Invalid JSON".to_string()));
/// ```
#[derive(Debug)]
pub enum AppError {
    /// Internal server errors (database failures, memory issues, etc.)
    Internal(String),
    
    /// Client sent invalid or malformed data
    BadRequest(String),
    
    /// Requested resource was not found
    NotFound(String),
    
    /// Configuration file or environment variable problems
    ConfigError(String),
    
    /// User input failed validation rules
    ValidationError(String),
}

/// Implementation of the Display trait for AppError.
/// 
/// ## Purpose:
/// This trait defines how errors are formatted as human-readable strings.
/// It's used when you print an error or convert it to a string.
/// 
/// ## Rust Concepts:
/// - **impl Trait for Type**: Implementing a trait for our custom type
/// - **match**: Pattern matching to handle each error variant
/// - **write!**: Macro for formatting strings (like printf in C)
/// - **&self**: Immutable reference to the error
/// 
/// ## Trait Requirements:
/// The Display trait requires one method: `fmt`. This method writes
/// the formatted error message to a formatter.
impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::Internal(msg) => write!(f, "Internal error: {}", msg),
            AppError::BadRequest(msg) => write!(f, "Bad request: {}", msg),
            AppError::NotFound(msg) => write!(f, "Not found: {}", msg),
            AppError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            AppError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
        }
    }
}

/// Implementation of the ResponseError trait for AppError.
/// 
/// ## Purpose:
/// This trait converts our custom errors into HTTP responses that clients can understand.
/// It automatically handles the conversion when an error is returned from a handler.
/// 
/// ## HTTP Status Code Mapping:
/// - Internal/ConfigError → 500 (Internal Server Error)
/// - BadRequest/ValidationError → 400 (Bad Request)
/// - NotFound → 404 (Not Found)
/// 
/// ## JSON Response Format:
/// All errors return JSON with a consistent structure:
/// ```json
/// {
///   "error": {
///     "type": "validation_error",
///     "message": "Port must be greater than 0",
///     "timestamp": "2025-01-01T12:00:00Z"
///   }
/// }
/// ```
/// 
/// ## Rust Concepts:
/// - **Tuple destructuring**: `let (a, b, c) = tuple`
/// - **json! macro**: Creates JSON values easily
/// - **StatusCode enum**: HTTP status codes as type-safe values
/// - **.clone()**: Creates a copy of the error message string
impl ResponseError for AppError {
    fn error_response(&self) -> HttpResponse {
        // Map each error type to HTTP status code, error type, and message
        let (status, error_type, message) = match self {
            AppError::Internal(msg) => (
                actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,  // 500
                "internal_error",
                msg.clone(),
            ),
            AppError::BadRequest(msg) => (
                actix_web::http::StatusCode::BAD_REQUEST,  // 400
                "bad_request",
                msg.clone(),
            ),
            AppError::NotFound(msg) => (
                actix_web::http::StatusCode::NOT_FOUND,  // 404
                "not_found",
                msg.clone(),
            ),
            AppError::ConfigError(msg) => (
                actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,  // 500
                "config_error",
                msg.clone(),
            ),
            AppError::ValidationError(msg) => (
                actix_web::http::StatusCode::BAD_REQUEST,  // 400
                "validation_error",
                msg.clone(),
            ),
        };

        // Build the HTTP response with JSON body
        HttpResponse::build(status).json(json!({
            "error": {
                "type": error_type,           // Machine-readable error type
                "message": message,           // Human-readable error message
                "timestamp": chrono::Utc::now().to_rfc3339()  // When the error occurred
            }
        }))
    }
}

/// Automatic conversion from anyhow::Error to AppError.
/// 
/// ## Purpose:
/// The anyhow crate provides general-purpose error handling. This conversion
/// allows us to use anyhow errors throughout the codebase and automatically
/// convert them to our custom error type when needed.
/// 
/// ## Rust Concepts:
/// - **From trait**: Enables automatic conversion with `.into()` or `?`
/// - **Self**: Refers to AppError (the type we're implementing for)
/// - **.to_string()**: Converts the error to a string representation
/// 
/// ## Usage:
/// When you use `?` with an anyhow::Error, it automatically becomes an AppError::Internal.
impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        AppError::Internal(err.to_string())
    }
}

/// Automatic conversion from JSON parsing errors to AppError.
/// 
/// ## Purpose:
/// When parsing JSON from HTTP requests fails, we want to return a BadRequest
/// error with a helpful message about what went wrong.
/// 
/// ## Why BadRequest:
/// JSON parsing errors are almost always due to the client sending malformed data,
/// so they should result in a 400 (Bad Request) response, not a 500 (Internal Server Error).
impl From<serde_json::Error> for AppError {
    fn from(err: serde_json::Error) -> Self {
        AppError::BadRequest(format!("JSON parsing error: {}", err))
    }
}

/// Automatic conversion from configuration errors to AppError.
/// 
/// ## Purpose:
/// Configuration loading can fail for various reasons (missing files, invalid syntax, etc.).
/// These are server-side issues that should be treated as configuration errors.
/// 
/// ## When this happens:
/// - config.toml file has invalid syntax
/// - Required environment variables are missing
/// - Configuration values fail validation
impl From<config::ConfigError> for AppError {
    fn from(err: config::ConfigError) -> Self {
        AppError::ConfigError(err.to_string())
    }
}

/// Type alias for Results that use our custom error type.
/// 
/// ## Purpose:
/// This creates a shorthand for `Result<T, AppError>` so you can write
/// `AppResult<String>` instead of `Result<String, AppError>`.
/// 
/// ## Usage Example:
/// ```rust
/// fn load_config() -> AppResult<AppConfig> {
///     // This is equivalent to: fn load_config() -> Result<AppConfig, AppError>
///     AppConfig::load()
/// }
/// ```
/// 
/// ## Rust Concepts:
/// - **type alias**: Creates a new name for an existing type
/// - **Generic type**: `T` can be any type (String, AppConfig, etc.)
pub type AppResult<T> = Result<T, AppError>;