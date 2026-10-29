//! High-quality noise cancellation library.
//!
//! Dual-engine design:
//! - **RNNoise** (default): ultra-lightweight, real-time friendly, pure Rust via `nnnoiseless`
//! - **DeepFilterNet3** (optional feature): higher quality full-band suppression via ONNX Runtime
//!
//! Designed for embedding into larger applications (audio studios, VoIP, DAWs, etc.).
//! Processes 48 kHz mono f32 audio in 10 ms frames (480 samples).

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod core;
pub mod engines;
pub mod processing;
pub mod error;

pub use error::{Error, Result};
pub use processing::{Denoiser, DenoiserConfig, EngineKind, Strength};

/// Re-export of the common sample type used throughout the library.
pub type Sample = f32;

/// Required sample rate for all engines (Hz).
pub const SAMPLE_RATE: u32 = 48_000;

/// Frame size used by both engines (samples @ 48 kHz = 10 ms).
pub const FRAME_SIZE: usize = 480;
