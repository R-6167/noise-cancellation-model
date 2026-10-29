//! High-level denoising API intended for application integration.

use crate::engines::NoiseEngine;
use crate::{Error, Result, FRAME_SIZE, SAMPLE_RATE};

#[cfg(feature = "rnnoise")]
use crate::engines::rnnoise::RnnoiseEngine;

/// Which engine to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    /// Ultra-lightweight RNNoise (default, always available when feature enabled).
    Rnnoise,
    /// Higher-quality DeepFilterNet3 (requires `deepfilternet` feature + model files).
    #[cfg(feature = "deepfilternet")]
    DeepFilterNet,
}

impl Default for EngineKind {
    fn default() -> Self {
        EngineKind::Rnnoise
    }
}

/// Suppression strength (0.0 = dry / no effect, 1.0 = full suppression).
#[derive(Debug, Clone, Copy)]
pub struct Strength(pub f32);

impl Strength {
    pub const MAX: Strength = Strength(1.0);
    pub const DEFAULT: Strength = Strength(0.85);

    pub fn clamp(self) -> Self {
        Strength(self.0.clamp(0.0, 1.0))
    }
}

impl Default for Strength {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Configuration for a Denoiser instance.
#[derive(Debug, Clone)]
pub struct DenoiserConfig {
    pub engine: EngineKind,
    pub strength: Strength,
    /// If true, the original (dry) signal is mixed back according to strength.
    pub wet_dry_mix: bool,
}

impl Default for DenoiserConfig {
    fn default() -> Self {
        Self {
            engine: EngineKind::default(),
            strength: Strength::DEFAULT,
            wet_dry_mix: true,
        }
    }
}

/// Main high-level denoiser.
///
/// One instance should be used per continuous audio stream.
/// Call `reset()` when switching to an unrelated recording.
pub struct Denoiser {
    engine: Box<dyn NoiseEngine>,
    config: DenoiserConfig,
    // Temporary buffers for wet/dry mixing
    dry_buf: Vec<f32>,
    wet_buf: Vec<f32>,
}

impl Denoiser {
    /// Create a new denoiser with the given configuration.
    pub fn new(config: DenoiserConfig) -> Result<Self> {
        let engine: Box<dyn NoiseEngine> = match config.engine {
            #[cfg(feature = "rnnoise")]
            EngineKind::Rnnoise => Box::new(RnnoiseEngine::new()),

            #[cfg(feature = "deepfilternet")]
            EngineKind::DeepFilterNet => {
                // Placeholder – full ONNX implementation will be added next.
                return Err(Error::FeatureDisabled(
                    "DeepFilterNet engine is not yet fully wired (coming in next iteration)",
                ));
            }

            #[cfg(not(feature = "rnnoise"))]
            EngineKind::Rnnoise => {
                return Err(Error::FeatureDisabled("rnnoise feature not enabled"));
            }
        };

        Ok(Self {
            engine,
            config,
            dry_buf: Vec::new(),
            wet_buf: Vec::new(),
        })
    }

    /// Convenience constructor with default settings (RNNoise, strength 0.85).
    pub fn with_defaults() -> Result<Self> {
        Self::new(DenoiserConfig::default())
    }

    /// Process a buffer of mono f32 samples at 48 kHz.
    ///
    /// The output buffer must be the same length as the input.
    /// This method is real-time safe when the engine is (RNNoise is).
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(Error::InvalidConfig(
                "input and output must have equal length".into(),
            ));
        }

        if input.is_empty() {
            return Ok(());
        }

        // Ensure work buffers are large enough
        if self.wet_buf.len() < input.len() {
            self.wet_buf.resize(input.len(), 0.0);
            self.dry_buf.resize(input.len(), 0.0);
        }

        // Run the engine
        self.engine.process(input, &mut self.wet_buf[..input.len()])?;

        let strength = self.config.strength.clamp().0;

        if self.config.wet_dry_mix && strength < 1.0 {
            // Mix dry + wet
            for i in 0..input.len() {
                output[i] = input[i] * (1.0 - strength) + self.wet_buf[i] * strength;
            }
        } else {
            output.copy_from_slice(&self.wet_buf[..input.len()]);
        }

        Ok(())
    }

    /// Reset internal engine state. Call when starting a new, unrelated stream.
    pub fn reset(&mut self) {
        self.engine.reset();
    }

    /// Current engine name.
    pub fn engine_name(&self) -> &'static str {
        self.engine.name()
    }

    /// Approximate algorithmic latency in samples.
    pub fn latency_samples(&self) -> usize {
        self.engine.latency_samples()
    }

    /// Update strength at runtime (cheap).
    pub fn set_strength(&mut self, strength: Strength) {
        self.config.strength = strength.clamp();
    }

    /// Current sample rate required by the library.
    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Preferred frame size (for lowest latency / best alignment).
    pub fn preferred_frame_size(&self) -> usize {
        FRAME_SIZE
    }
}
