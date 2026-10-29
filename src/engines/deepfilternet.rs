//! DeepFilterNet3 engine (ONNX Runtime).
//!
//! This module is a placeholder for the high-quality engine.
//! Full implementation will load the official DeepFilterNet3 ONNX graphs
//! and perform streaming inference with proper state management.

#![cfg(feature = "deepfilternet")]

use crate::engines::NoiseEngine;
use crate::{Error, Result};

pub struct DeepFilterNetEngine {
    // Future fields: ort::Session, STFT state, ERB filters, etc.
}

impl DeepFilterNetEngine {
    pub fn new(_model_dir: &std::path::Path) -> Result<Self> {
        Err(Error::Engine(
            "DeepFilterNet3 engine not yet implemented – coming soon".into(),
        ))
    }
}

impl NoiseEngine for DeepFilterNetEngine {
    fn process(&mut self, _input: &[f32], _output: &mut [f32]) -> Result<()> {
        Err(Error::Engine("not implemented".into()))
    }

    fn reset(&mut self) {}

    fn name(&self) -> &'static str {
        "DeepFilterNet3"
    }

    fn latency_samples(&self) -> usize {
        960 // typical ~20 ms
    }
}
