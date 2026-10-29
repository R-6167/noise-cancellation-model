//! High-quality noise cancellation library.
//!
//! Dual-engine design:
//! - **RNNoise** (default): ultra-lightweight, real-time friendly, pure Rust via `nnnoiseless`.
//!   Fully self-contained — weights are embedded; no files or network required.
//! - **DeepFilterNet3** (optional feature): higher quality full-band suppression via ONNX Runtime.
//!   Uses only local model files; the library never downloads anything at runtime.
//!
//! C ABI: enable feature `c-api` and include `include/noise_cancellation.h`.
//! Sample rates other than 48 kHz are resampled transparently.
//!
//! Designed for embedding into larger applications (audio studios, VoIP, DAWs, etc.).

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod core;
pub mod engines;
pub mod processing;
pub mod error;
pub mod utils;

#[cfg(feature = "c-api")]
pub mod c_api;

pub use error::{Error, Result};
pub use processing::{Denoiser, DenoiserConfig, EngineKind, Strength};
pub use utils::{
    AudioBufferPool, LinearResampler, PooledBuffer, default_model_dir, find_model_dir, is_model_dir,
};

/// Re-export of the common sample type used throughout the library.
pub type Sample = f32;

/// Engine internal sample rate (Hz). Host rates are converted to/from this.
pub const SAMPLE_RATE: u32 = 48_000;

/// Frame size at engine rate (samples @ 48 kHz = 10 ms).
pub const FRAME_SIZE: usize = 480;
