//! Error types for the noise cancellation library.

use thiserror::Error;

/// Library-wide error type.
#[derive(Debug, Error)]
pub enum Error {
    /// Invalid configuration or parameters.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Audio format / sample-rate mismatch.
    #[error("unsupported audio format: {0}")]
    UnsupportedFormat(String),

    /// Engine-specific failure (model load, inference, etc.).
    #[error("engine error: {0}")]
    Engine(String),

    /// I/O related error (mainly for CLI / offline path).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Feature not compiled in.
    #[error("feature not enabled: {0}")]
    FeatureDisabled(&'static str),
}

/// Convenient Result alias.
pub type Result<T> = std::result::Result<T, Error>;
