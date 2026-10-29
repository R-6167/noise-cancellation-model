//! High-level denoising API intended for application integration.

mod multi;
pub use multi::{MultiDenoiser, MAX_CHANNELS};

use crate::engines::NoiseEngine;
#[cfg(all(not(feature = "rnnoise"), feature = "deepfilternet"))]
use crate::utils::default_model_dir;
use crate::utils::LinearResampler;
use std::collections::VecDeque;
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

#[allow(clippy::derivable_impls)]
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
        if self.0.is_finite() {
            Strength(self.0.clamp(0.0, 1.0))
        } else {
            Strength(0.0)
        }
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
    dry_queue: VecDeque<f32>,
    dry_delay_samples: usize,
    flush_queue: VecDeque<f32>,
    flushing: bool,
    flush_remaining: usize,
}

impl Denoiser {
    pub fn new(config: DenoiserConfig) -> Result<Self> {
        if config.sample_rate == 0 {
            return Err(Error::InvalidConfig("sample_rate must be > 0".into()));
        }

        let mut engine: Box<dyn NoiseEngine> = match &config.engine {
            #[cfg(feature = "rnnoise")]
            EngineKind::Rnnoise => Box::new(RnnoiseEngine::new()),

            #[cfg(feature = "deepfilternet")]
            EngineKind::DeepFilterNet { model_dir } => {
                let df_config = DeepFilterNetConfig::new(model_dir.clone());
                Box::new(DeepFilterNetEngine::new(df_config)?)
            }
        };

        engine.set_strength(config.strength.clamp().0);
        let dry_delay_samples = ((engine.latency_samples() as u64 * config.sample_rate as u64)
            / SAMPLE_RATE as u64) as usize;
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
            dry_queue: std::iter::repeat_n(0.0, dry_delay_samples).collect(),
            dry_delay_samples,
            flush_queue: VecDeque::new(),
            flushing: false,
            flush_remaining: 0,
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
        if self.flushing {
            return Err(Error::InvalidConfig("stream has been flushed; call reset before processing more audio".into()));
        }
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
            let result = self.process_native(input, output);
            if result.is_ok() {
                sanitize_slice(output);
            }
            return result;
        }

        self.up.as_mut().unwrap().process(input, &mut self.engine_in);
        let strength = self.config.strength.clamp().0;

        if self.engine_in.is_empty() {
            self.dry_queue.extend(input.iter().copied());
            for (i, sample) in output.iter_mut().enumerate() {
                let dry = self.dry_queue.pop_front().unwrap_or(0.0);
                *sample = if self.config.wet_dry_mix && strength == 0.0 {
                    // A fully dry setting is a true pass-through, without algorithmic latency.
                    input[i]
                } else if self.config.wet_dry_mix {
                    dry * (1.0 - strength)
                } else {
                    0.0
                };
            }
            sanitize_slice(output);
            return Ok(());
        }

        if self.engine_out.len() < self.engine_in.len() {
            self.engine_out.resize(self.engine_in.len(), 0.0);
        }
        self.engine.process(
            &self.engine_in,
            &mut self.engine_out[..self.engine_in.len()],
        )?;

        self.dry_queue.extend(input.iter().copied());
        let down = self.down.as_mut().unwrap();
        let mut wet_host = std::mem::take(&mut self.wet_buf);
        down.process(&self.engine_out[..self.engine_in.len()], &mut wet_host);
        for (i, sample) in output.iter_mut().enumerate() {
            let dry = self.dry_queue.pop_front().unwrap_or(0.0);
            let wet = wet_host.get(i).copied().unwrap_or(0.0);
            *sample = if self.config.wet_dry_mix && strength == 0.0 {
                input[i]
            } else if self.config.wet_dry_mix {
                dry * (1.0 - strength) + wet * strength
            } else {
                wet
            };
        }
        self.wet_buf = wet_host;

        sanitize_slice(output);
        Ok(())
    }

    fn process_native(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if self.wet_buf.len() < input.len() {
            self.wet_buf.resize(input.len(), 0.0);
        }
        self.engine
            .process(input, &mut self.wet_buf[..input.len()])?;
        let strength = self.config.strength.clamp().0;
        self.dry_queue.extend(input.iter().copied());
        for (i, sample) in output.iter_mut().enumerate() {
            let dry = self.dry_queue.pop_front().unwrap_or(0.0);
            let wet = self.wet_buf[i];
            *sample = if self.config.wet_dry_mix && strength == 0.0 {
                input[i]
            } else if self.config.wet_dry_mix {
                dry * (1.0 - strength) + wet * strength
            } else {
                wet
            };
        }
        Ok(())
    }

    /// Finish the stream and drain the remaining delayed output.
    ///
    /// Returns the number of valid samples written. Call repeatedly with an
    /// output buffer of any size until this returns `0`. After flushing,
    /// call `reset` before processing another stream.
    pub fn flush(&mut self, output: &mut [f32]) -> Result<usize> {
        if output.is_empty() || (self.flushing && self.flush_remaining == 0) {
            return Ok(0);
        }

        if !self.flushing {
            self.flushing = true;
            self.flush_remaining = self.latency_samples();

            let engine_latency = self.engine.latency_samples();
            let mut engine_tail = vec![0.0f32; engine_latency.max(1)];
            let written = self.engine.flush(&mut engine_tail)?;
            engine_tail.truncate(written);

            let mut wet_host = Vec::new();
            if let Some(down) = self.down.as_mut() {
                down.process(&engine_tail, &mut wet_host);
            } else {
                wet_host.extend_from_slice(&engine_tail);
            }

            let target = self.flush_remaining;
            wet_host.resize(target, 0.0);
            wet_host.truncate(target);
            self.flush_queue.extend(wet_host);
        }

        let count = output.len().min(self.flush_remaining);
        let strength = self.config.strength.clamp().0;
        for sample in output.iter_mut().take(count) {
            let wet = self.flush_queue.pop_front().unwrap_or(0.0);
            let dry = self.dry_queue.pop_front().unwrap_or(0.0);
            *sample = if self.config.wet_dry_mix && strength == 0.0 {
                0.0
            } else if self.config.wet_dry_mix {
                dry * (1.0 - strength) + wet * strength
            } else {
                wet
            };
        }
        self.flush_remaining -= count;
        if self.flush_remaining == 0 {
            self.flush_queue.clear();
        }
        sanitize_slice(&mut output[..count]);
        Ok(count)
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
        self.dry_queue.clear();
        self.dry_queue.extend(std::iter::repeat_n(0.0, self.dry_delay_samples));
        self.flush_queue.clear();
        self.flushing = false;
        self.flush_remaining = 0;
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
        self.engine.set_strength(self.config.strength.0);
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
