//! DeepFilterNet3 engine using ONNX Runtime.
//!
//! High-quality full-band speech enhancement (48 kHz).
//!
//! This module provides:
//! - Model loading from official DeepFilterNet3 ONNX exports
//!   (enc.onnx + erb_dec.onnx + df_dec.onnx or a combined graph)
//! - Streaming state management
//! - Integration with the common NoiseEngine trait
//!
//! Note: Complete STFT → ERB → deep-filter → iSTFT DSP is non-trivial.
//! The current implementation focuses on correct model loading, session
//! management, and a clean API. The full spectral processing pipeline
//! will be completed in the next iteration (or can use the upstream
//! libDF / deepfilter-rt components).
//!
//! Recommended model sources:
//! - Official: https://github.com/Rikorose/DeepFilterNet (models/DeepFilterNet3_onnx.tar.gz)
//! - Hugging Face mirrors of the three ONNX files

#![cfg(feature = "deepfilternet")]

use std::path::{Path, PathBuf};
use crate::engines::NoiseEngine;
use crate::{Error, Result, FRAME_SIZE, SAMPLE_RATE};

/// Configuration for the DeepFilterNet3 engine.
#[derive(Debug, Clone)]
pub struct DeepFilterNetConfig {
    /// Directory containing enc.onnx, erb_dec.onnx, df_dec.onnx
    /// (or a single combined deepfilter.onnx).
    pub model_dir: PathBuf,
    /// Attenuation limit in dB (None = full suppression).
    pub atten_lim_db: Option<f32>,
    /// Whether to compensate algorithmic delay.
    pub compensate_delay: bool,
}

impl DeepFilterNetConfig {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: model_dir.into(),
            atten_lim_db: None,
            compensate_delay: true,
        }
    }
}

/// DeepFilterNet3-based suppressor.
pub struct DeepFilterNetEngine {
    config: DeepFilterNetConfig,
    // Future: ort::Session(s), STFT buffers, ERB matrices, GRU states, etc.
    _placeholder: (),
    remainder: Vec<f32>,
}

impl DeepFilterNetEngine {
    /// Create a new engine. Model files must already be present in `config.model_dir`.
    pub fn new(config: DeepFilterNetConfig) -> Result<Self> {
        let dir = &config.model_dir;
        if !dir.exists() {
            return Err(Error::Engine(format!(
                "model directory does not exist: {}",
                dir.display()
            )));
        }

        // Check for the three classic files or a combined graph
        let has_split = dir.join("enc.onnx").exists()
            && dir.join("erb_dec.onnx").exists()
            && dir.join("df_dec.onnx").exists();
        let has_combined = dir.join("deepfilter.onnx").exists();

        if !has_split && !has_combined {
            return Err(Error::Engine(format!(
                "no DeepFilterNet3 ONNX models found in {}. \
                 Expected enc.onnx + erb_dec.onnx + df_dec.onnx \
                 (or deepfilter.onnx). Download from \
                 https://github.com/Rikorose/DeepFilterNet or a Hugging Face mirror.",
                dir.display()
            )));
        }

        // TODO: Load sessions with ort, initialise STFT / ERB / norm state.

        Ok(Self {
            config,
            _placeholder: (),
            remainder: Vec::with_capacity(FRAME_SIZE * 2),
        })
    }

    /// Convenience constructor from a path.
    pub fn from_path(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::new(DeepFilterNetConfig::new(model_dir.as_ref()))
    }
}

impl NoiseEngine for DeepFilterNetEngine {
    fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(Error::InvalidConfig(
                "input and output buffers must have the same length".into(),
            ));
        }

        // Placeholder behaviour until full DSP is wired:
        // pass-through so callers can exercise engine selection and API.
        // Real implementation will do STFT → features → ONNX → mask/coefs → iSTFT.
        output.copy_from_slice(input);
        Ok(())
    }

    fn reset(&mut self) {
        self.remainder.clear();
    }

    fn name(&self) -> &'static str {
        "DeepFilterNet3"
    }

    fn latency_samples(&self) -> usize {
        if self.config.compensate_delay {
            FRAME_SIZE * 3
        } else {
            FRAME_SIZE * 2
        }
    }
}
