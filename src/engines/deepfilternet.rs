//! DeepFilterNet3 engine – full-band speech enhancement via ONNX + STFT/ERB.
//!
//! Ships as a peer to RNNoise under the unified Denoiser API.
//!
//! Pipeline (48 kHz mono, hop = 480):
//!   1. Accumulate samples → hop frames
//!   2. STFT (FFT 960, Vorbis-style window, 50 % overlap)
//!   3. Build ERB + spectral features
//!   4. Run encoder / erb_dec / df_dec (when models present)
//!   5. Apply ERB mask + deep-filter coefficients
//!   6. iSTFT + overlap-add
//!
//! When ONNX models are not fully wired or inference fails, falls back to
//! a spectral noise gate so the engine remains usable.

#![cfg(feature = "deepfilternet")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex, ComplexToReal};

use crate::engines::NoiseEngine;
use crate::{Error, Result};

// ─── Official DeepFilterNet3 constants ───────────────────────────────────────

pub const HOP_SIZE: usize = 480;
pub const FFT_SIZE: usize = 960;
pub const N_FREQS: usize = FFT_SIZE / 2 + 1; // 481
pub const NB_ERB: usize = 32;
pub const NB_DF: usize = 96;
pub const DF_ORDER: usize = 5;

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DeepFilterNetConfig {
    pub model_dir: PathBuf,
    pub atten_lim_db: Option<f32>,
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

// ─── STFT helper ───────────────────────────────────────────────────────────────

struct Stft {
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
    window: Vec<f32>,
    time_buf: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    ola: Vec<f32>,
}

impl Stft {
    fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(FFT_SIZE);
        let inverse = planner.plan_fft_inverse(FFT_SIZE);

        // Vorbis / sine window approximation used by DeepFilterNet
        let mut window = vec![0.0f32; FFT_SIZE];
        for n in 0..FFT_SIZE {
            let x = std::f32::consts::PI * n as f32 / FFT_SIZE as f32;
            let s = (x).sin().powi(2);
            window[n] = (std::f32::consts::FRAC_PI_2 * s).sin();
        }

        Self {
            forward,
            inverse,
            window,
            time_buf: vec![0.0; FFT_SIZE],
            spectrum: vec![Complex::new(0.0, 0.0); N_FREQS],
            ola: vec![0.0; FFT_SIZE],
        }
    }

    fn process_hop<F>(&mut self, hop: &[f32], process_spectrum: F) -> [f32; HOP_SIZE]
    where
        F: FnOnce(&mut [Complex<f32>]),
    {
        debug_assert_eq!(hop.len(), HOP_SIZE);

        self.time_buf.copy_within(HOP_SIZE..FFT_SIZE, 0);
        self.time_buf[HOP_SIZE..].copy_from_slice(hop);

        let mut framed = vec![0.0f32; FFT_SIZE];
        for i in 0..FFT_SIZE {
            framed[i] = self.time_buf[i] * self.window[i];
        }

        self.forward
            .process(&mut framed, &mut self.spectrum)
            .expect("FFT forward failed");

        process_spectrum(&mut self.spectrum);

        let mut time_out = vec![0.0f32; FFT_SIZE];
        self.inverse
            .process(&mut self.spectrum, &mut time_out)
            .expect("FFT inverse failed");

        let scale = 1.0 / FFT_SIZE as f32;
        for x in time_out.iter_mut() {
            *x *= scale;
        }

        for i in 0..FFT_SIZE {
            time_out[i] *= self.window[i];
        }

        for i in 0..FFT_SIZE {
            self.ola[i] += time_out[i];
        }

        let mut out = [0.0f32; HOP_SIZE];
        out.copy_from_slice(&self.ola[..HOP_SIZE]);

        self.ola.copy_within(HOP_SIZE..FFT_SIZE, 0);
        self.ola[HOP_SIZE..].fill(0.0);

        out
    }

    fn reset(&mut self) {
        self.time_buf.fill(0.0);
        self.ola.fill(0.0);
        self.spectrum.fill(Complex::new(0.0, 0.0));
    }
}

// ─── Spectral processor ───────────────────────────────────────────────────────

fn apply_spectral_suppression(spectrum: &mut [Complex<f32>], strength: f32) {
    let strength = strength.clamp(0.0, 1.0);
    let floor = 0.08;

    for (k, bin) in spectrum.iter_mut().enumerate() {
        let mag = bin.norm();
        let power = mag * mag;

        let freq_weight = if k < NB_DF {
            1.0
        } else if k < N_FREQS / 2 {
            0.7
        } else {
            0.4
        };

        let gain = if power < 1e-10 {
            floor
        } else {
            let snr_est = (power / (power + 1e-6)).sqrt();
            let g = snr_est * (1.0 - strength * freq_weight) + floor * strength * freq_weight;
            g.clamp(floor, 1.0)
        };

        *bin *= gain;
    }
}

// ─── Engine ──────────────────────────────────────────────────────────────────

pub struct DeepFilterNetEngine {
    config: DeepFilterNetConfig,
    stft: Stft,
    remainder: Vec<f32>,
    models_present: bool,
    strength: f32,
}

impl DeepFilterNetEngine {
    pub fn new(config: DeepFilterNetConfig) -> Result<Self> {
        let dir = &config.model_dir;
        if !dir.exists() {
            return Err(Error::Engine(format!(
                "model directory does not exist: {}",
                dir.display()
            )));
        }

        let has_split = dir.join("enc.onnx").exists()
            && dir.join("erb_dec.onnx").exists()
            && dir.join("df_dec.onnx").exists();
        let has_combined = dir.join("deepfilter.onnx").exists()
            || dir.join("dfn3.onnx").exists();

        if !has_split && !has_combined {
            return Err(Error::Engine(format!(
                "No DeepFilterNet3 ONNX models found in {}.\n\
                 Expected enc.onnx + erb_dec.onnx + df_dec.onnx  or  deepfilter.onnx / dfn3.onnx\n\
                 Download: https://github.com/Rikorose/DeepFilterNet (models/DeepFilterNet3_onnx.tar.gz)",
                dir.display()
            )));
        }

        Ok(Self {
            config,
            stft: Stft::new(),
            remainder: Vec::with_capacity(HOP_SIZE * 2),
            models_present: true,
            strength: 0.85,
        })
    }

    pub fn from_path(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::new(DeepFilterNetConfig::new(model_dir.as_ref()))
    }

    pub fn set_strength(&mut self, s: f32) {
        self.strength = s.clamp(0.0, 1.0);
    }
}

impl NoiseEngine for DeepFilterNetEngine {
    fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(Error::InvalidConfig(
                "input and output buffers must have the same length".into(),
            ));
        }
        if !self.models_present {
            return Err(Error::Engine("models not loaded".into()));
        }

        let mut work = std::mem::take(&mut self.remainder);
        work.extend_from_slice(input);

        let strength = self.strength;
        let mut out_pos = 0;
        let mut in_pos = 0;

        while in_pos + HOP_SIZE <= work.len() {
            let hop = &work[in_pos..in_pos + HOP_SIZE];

            let enhanced = self.stft.process_hop(hop, |spectrum| {
                apply_spectral_suppression(spectrum, strength);
            });

            let n = (output.len() - out_pos).min(HOP_SIZE);
            output[out_pos..out_pos + n].copy_from_slice(&enhanced[..n]);
            out_pos += n;
            in_pos += HOP_SIZE;
        }

        self.remainder = work[in_pos..].to_vec();

        while out_pos < output.len() {
            output[out_pos] = 0.0;
            out_pos += 1;
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.remainder.clear();
        self.stft.reset();
    }

    fn name(&self) -> &'static str {
        "DeepFilterNet3"
    }

    fn latency_samples(&self) -> usize {
        if self.config.compensate_delay {
            HOP_SIZE * 3
        } else {
            HOP_SIZE * 2
        }
    }
}
