//! High-level denoising API intended for application integration.

mod multi;
pub use multi::{MultiDenoiser, MAX_CHANNELS};

use crate::engines::NoiseEngine;
use crate::utils::{default_model_dir, LinearResampler};
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
    /// Higher-quality DeepFilterNet3 (requires `deepfilternet` feature + local model files).
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
                model_dir: default_model_dir(),
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
    /// Host sample rate in Hz (engines run at 48 kHz with transparent resampling).
    pub sample_rate: u32,
}

impl Default for DenoiserConfig {
    fn default() -> Self {
        Self {
            engine: EngineKind::default(),
            strength: Strength::DEFAULT,
            wet_dry_mix: true,
            sample_rate: SAMPLE_RATE,
        }
    }
}

fn sanitize_slice(buf: &mut [f32]) {
    for x in buf.iter_mut() {
        if !x.is_finite() {
            *x = 0.0;
        }
    }
}

/// Main high-level mono denoiser.
pub struct Denoiser {
    engine: Box<dyn NoiseEngine>,
    config: DenoiserConfig,
    wet_buf: Vec<f32>,
    up: Option<LinearResampler>,
    down: Option<LinearResampler>,
    engine_in: Vec<f32>,
    engine_out: Vec<f32>,
}

impl Denoiser {
    pub fn new(config: DenoiserConfig) -> Result<Self> {
        if config.sample_rate == 0 {
            return Err(Error::InvalidConfig("sample_rate must be > 0".into()));
        }

        let engine: Box<dyn NoiseEngine> = match &config.engine {
            #[cfg(feature = "rnnoise")]
            EngineKind::Rnnoise => Box::new(RnnoiseEngine::new()),

            #[cfg(feature = "deepfilternet")]
            EngineKind::DeepFilterNet { model_dir } => {
                let df_config = DeepFilterNetConfig::new(model_dir.clone());
                Box::new(DeepFilterNetEngine::new(df_config)?)
            }
        };

        let (up, down) = if config.sample_rate != SAMPLE_RATE {
            (
                Some(LinearResampler::new(config.sample_rate, SAMPLE_RATE)),
                Some(LinearResampler::new(SAMPLE_RATE, config.sample_rate)),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            engine,
            config,
            wet_buf: Vec::new(),
            up,
            down,
            engine_in: Vec::new(),
            engine_out: Vec::new(),
        })
    }

    pub fn with_defaults() -> Result<Self> {
        Self::new(DenoiserConfig::default())
    }

    pub fn with_sample_rate(sample_rate: u32) -> Result<Self> {
        Self::new(DenoiserConfig {
            sample_rate,
            ..Default::default()
        })
    }

    #[cfg(feature = "deepfilternet")]
    pub fn with_deepfilternet(model_dir: impl Into<std::path::PathBuf>) -> Result<Self> {
        Self::new(DenoiserConfig {
            engine: EngineKind::DeepFilterNet {
                model_dir: model_dir.into(),
            },
            ..Default::default()
        })
    }

    #[cfg(feature = "deepfilternet")]
    pub fn with_deepfilternet_auto() -> Result<Self> {
        let model_dir = crate::utils::find_model_dir().ok_or_else(|| {
            Error::Engine(
                "DeepFilterNet3 models not found. Place enc.onnx, erb_dec.onnx, df_dec.onnx \
                 in ./models or set NOISE_CANCEL_MODEL_DIR."
                    .into(),
            )
        })?;
        Self::with_deepfilternet(model_dir)
    }

    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(Error::InvalidConfig(
                "input and output must have equal length".into(),
            ));
        }
        if input.is_empty() {
            return Ok(());
        }
        if input.len() > 48_000 * 30 {
            return Err(Error::InvalidConfig(
                "block too large (max 30s at 48 kHz host samples)".into(),
            ));
        }

        if self.up.is_none() {
            let r = self.process_native(input, output);
            if r.is_ok() {
                sanitize_slice(output);
            }
            return r;
        }

        let up = self.up.as_mut().unwrap();
        up.process(input, &mut self.engine_in);
        if self.engine_in.is_empty() {
            output.fill(0.0);
            return Ok(());
        }

        if self.engine_out.len() < self.engine_in.len() {
            self.engine_out.resize(self.engine_in.len(), 0.0);
        }
        self.engine
            .process(&self.engine_in, &mut self.engine_out[..self.engine_in.len()])?;

        let strength = self.config.strength.clamp().0;
        if self.config.wet_dry_mix && strength < 1.0 {
            let down = self.down.as_mut().unwrap();
            let mut wet_host = std::mem::take(&mut self.wet_buf);
            down.process(&self.engine_out[..self.engine_in.len()], &mut wet_host);
            let n = input.len().min(wet_host.len());
            for i in 0..n {
                output[i] = input[i] * (1.0 - strength) + wet_host[i] * strength;
            }
            for i in n..output.len() {
                output[i] = input[i];
            }
            self.wet_buf = wet_host;
        } else {
            let down = self.down.as_mut().unwrap();
            let mut wet_host = std::mem::take(&mut self.wet_buf);
            down.process(&self.engine_out[..self.engine_in.len()], &mut wet_host);
            let n = input.len().min(wet_host.len());
            output[..n].copy_from_slice(&wet_host[..n]);
            for i in n..output.len() {
                output[i] = wet_host.last().copied().unwrap_or(0.0);
            }
            self.wet_buf = wet_host;
        }

        sanitize_slice(output);
        Ok(())
    }

    fn process_native(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
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
        if let Some(r) = self.up.as_mut() {
            r.reset();
        }
        if let Some(r) = self.down.as_mut() {
            r.reset();
        }
        self.engine_in.clear();
        self.engine_out.clear();
    }

    pub fn engine_name(&self) -> &'static str {
        self.engine.name()
    }

    pub fn latency_samples(&self) -> usize {
        let eng = self.engine.latency_samples();
        if self.config.sample_rate == SAMPLE_RATE {
            eng
        } else {
            ((eng as u64 * self.config.sample_rate as u64) / SAMPLE_RATE as u64) as usize
        }
    }

    pub fn set_strength(&mut self, strength: Strength) {
        self.config.strength = strength.clamp();
    }

    pub fn strength(&self) -> Strength {
        self.config.strength
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    pub fn preferred_frame_size(&self) -> usize {
        let num = FRAME_SIZE as u64 * self.config.sample_rate as u64;
        ((num + SAMPLE_RATE as u64 / 2) / SAMPLE_RATE as u64) as usize
    }
}
