//! High-quality noise cancellation library.
//!
//! Dual-engine design:
//! - **RNNoise** (default): ultra-lightweight, embedded weights.
//! - **DeepFilterNet3** (optional): higher quality, local ONNX only.
//!
//! C ABI, multi-rate resampling, multi-channel (`MultiDenoiser`).

#![deny(unsafe_op_in_unsafe_fn)]

pub mod core;
pub mod engines;
pub mod processing;
pub mod error;
pub mod utils;

#[cfg(feature = "c-api")]
pub mod c_api;

pub use error::{Error, Result};
pub use processing::{
    Denoiser, DenoiserConfig, EngineKind, MultiDenoiser, Strength, MAX_CHANNELS,
};
pub use utils::{
    AudioBufferPool, LinearResampler, PooledBuffer, default_model_dir, find_model_dir, is_model_dir,
};

/// Common sample type.
pub type Sample = f32;

/// Engine internal sample rate (Hz).
pub const SAMPLE_RATE: u32 = 48_000;

/// Frame size at engine rate (10 ms @ 48 kHz).
pub const FRAME_SIZE: usize = 480;
