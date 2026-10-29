//! High-level denoising API intended for application integration.

use crate::engines::NoiseEngine;
use crate::{Error, Result, FRAME_SIZE, SAMPLE_RATE};

#[cfg(feature = "rnnoise")]
use crate::engines::rnnoise::RnnoiseEngine;

#[cfg(feature = "deepfilternet")]
use crate::engines::deepfilternet::{DeepFilterNetConfig, DeepFilterNetEngine};

/// Which engine to use.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineKind {
    /// Ultra-lightweight RNNoise (default when feature enabled).
    #[cfg(feature = "rnnoise")]
    Rnnoise,
    /// Higher-quality DeepFilterNet3 (requires `deepfilternet` feature + model files).
    #[cfg(feature = "deepfilternet")]
    DeepFilterNet {
        /// Directory containing the ONNX model files.
        model_dir: std::path::PathBuf,
    },
}

impl Default for EngineKind {
    fn default() -> Self {
        #[cfg(feature = "rnnoise")]
        {
            EngineKind::Rnnoise
        }
        #[cfg(all(not(feature = "rnnoise"), feature = "deepfilternet"))]
        {
            EngineKind::DeepFilterNet {
                model_dir: std::path::PathBuf::from("models"),
            }
        }
        #[cfg(all(not(feature = "rnnoise"), not(feature = "deepfilternet")))]
        compile_error!("At least one of the features `rnnoise` or `deepfilternet` must be enabled");
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
    wet_buf: Vec<f32>,
}

impl Denoiser {
    /// Create a new denoiser with the given configuration.
    pub fn new(config: DenoiserConfig) -> Result<Self> {
        let engine: Box<dyn NoiseEngine> = match &config.engine {
            #[cfg(feature = "rnnoise")]
            EngineKind::Rnnoise => Box::new(RnnoiseEngine::new()),

            #[cfg(feature = "deepfilternet")]
            EngineKind::DeepFilterNet { model_dir } => {
                let df_config = DeepFilterNetConfig::new(model_dir.clone());
                Box::new(DeepFilterNetEngine::new(df_config)?)
            }
        };

        Ok(Self {
            engine,
            config,
            wet_buf: Vec::new(),
        })
    }

    /// Convenience constructor with default settings (RNNoise when available).
    pub fn with_defaults() -> Result<Self> {
        Self::new(DenoiserConfig::default())
    }

    /// Create a Denoiser that uses DeepFilterNet3.
    #[cfg(feature = "deepfilternet")]
    pub fn with_deepfilternet(model_dir: impl Into<std::path::PathBuf>) -> Result<Self> {
        let config = DenoiserConfig {
            engine: EngineKind::DeepFilterNet {
                model_dir: model_dir.into(),
            },
            ..Default::default()
        };
        Self::new(config)
    }

    /// Process a buffer of mono f32 samples at 48 kHz.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(Error::InvalidConfig(
                "input and output must have equal length".into(),
            ));
        }

        if input.is_empty() {
            return Ok(());
        }

        if self.wet_buf.len() < input.len() {
            self.wet_buf.resize(input.len(), 0.0);
        }

        self.engine.process(input, &mut self.wet_buf[..input.len()])?;

        let strength = self.config.strength.clamp().0;

        if self.config.wet_dry_mix && strength < 1.0 {
            for i in 0..input.len() {
                output[i] = input[i] * (1.0 - strength) + self.wet_buf[i] * strength;
            }
        } else {
            output.copy_from_slice(&self.wet_buf[..input.len()]);
        }

        Ok(())
    }

    pub fn reset(&mut self) {
        self.engine.reset();
    }

    pub fn engine_name(&self) -> &'static str {
        self.engine.name()
    }

    pub fn latency_samples(&self) -> usize {
        self.engine.latency_samples()
    }

    pub fn set_strength(&mut self, strength: Strength) {
        self.config.strength = strength.clamp();
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    pub fn preferred_frame_size(&self) -> usize {
        FRAME_SIZE
    }
}
