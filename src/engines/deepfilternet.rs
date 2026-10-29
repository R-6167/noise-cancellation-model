//! DeepFilterNet3 engine – STFT + ONNX encoder inference.
//!
//! Loads enc / erb_dec / df_dec (or combined) via ort, keeps sessions alive,
//! builds feat_erb + feat_spec from the STFT, and runs the encoder each hop.
//! Falls back to spectral suppression when neural path is unavailable.

#![cfg(feature = "deepfilternet")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::engines::NoiseEngine;
use crate::{Error, Result};

// ─── Constants ──────────────────────────────────────────────────────────────

pub const HOP_SIZE: usize = 480;
pub const FFT_SIZE: usize = 960;
pub const N_FREQS: usize = FFT_SIZE / 2 + 1; // 481
pub const NB_ERB: usize = 32;
pub const NB_DF: usize = 96;
pub const DF_ORDER: usize = 5;

// ─── Config ─────────────────────────────────────────────────────────────────

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

// ─── STFT ───────────────────────────────────────────────────────────────────

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

        let mut window = vec![0.0f32; FFT_SIZE];
        for n in 0..FFT_SIZE {
            let x = std::f32::consts::PI * n as f32 / FFT_SIZE as f32;
            let s = x.sin().powi(2);
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
        self.time_buf.copy_within(HOP_SIZE..FFT_SIZE, 0);
        self.time_buf[HOP_SIZE..].copy_from_slice(hop);

        let mut framed = vec![0.0f32; FFT_SIZE];
        for i in 0..FFT_SIZE {
            framed[i] = self.time_buf[i] * self.window[i];
        }

        self.forward
            .process(&mut framed, &mut self.spectrum)
            .expect("FFT forward");

        process_spectrum(&mut self.spectrum);

        let mut time_out = vec![0.0f32; FFT_SIZE];
        self.inverse
            .process(&mut self.spectrum, &mut time_out)
            .expect("FFT inverse");

        let scale = 1.0 / FFT_SIZE as f32;
        for x in time_out.iter_mut() {
            *x *= scale;
        }
        for i in 0..FFT_SIZE {
            time_out[i] *= self.window[i];
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

// ─── ERB / spectral features ────────────────────────────────────────────

fn compute_erb_features(spectrum: &[Complex<f32>]) -> [f32; NB_ERB] {
    let mut erb = [0.0f32; NB_ERB];
    let n_freqs = spectrum.len().min(N_FREQS);
    let max_bin = NB_DF.min(n_freqs);
    for b in 0..NB_ERB {
        let lo = (b as f32 / NB_ERB as f32 * max_bin as f32) as usize;
        let hi = (((b + 1) as f32 / NB_ERB as f32) * max_bin as f32) as usize;
        let hi = hi.max(lo + 1).min(n_freqs);
        let mut energy = 0.0f32;
        for k in lo..hi {
            energy += spectrum[k].norm_sqr();
        }
        erb[b] = (energy + 1e-10).ln();
    }
    erb
}

fn compute_spec_features(spectrum: &[Complex<f32>]) -> ([f32; NB_DF], [f32; NB_DF]) {
    let mut re = [0.0f32; NB_DF];
    let mut im = [0.0f32; NB_DF];
    for k in 0..NB_DF.min(spectrum.len()) {
        re[k] = spectrum[k].re;
        im[k] = spectrum[k].im;
    }
    (re, im)
}

// ─── ONNX models ─────────────────────────────────────────────────────────

struct OnnxModels {
    ready: bool,
    enc: Option<ort::session::Session>,
    loaded_paths: Vec<PathBuf>,
}

impl OnnxModels {
    fn try_load(model_dir: &Path) -> Result<Self> {
        let enc_path = model_dir.join("enc.onnx");
        let erb_path = model_dir.join("erb_dec.onnx");
        let df_path = model_dir.join("df_dec.onnx");
        let combined = model_dir.join("deepfilter.onnx");
        let combined2 = model_dir.join("dfn3.onnx");

        let has_split = enc_path.exists() && erb_path.exists() && df_path.exists();
        let has_combined = combined.exists() || combined2.exists();

        if !has_split && !has_combined {
            return Err(Error::Engine(format!(
                "No DeepFilterNet3 ONNX models found in {}.\n\
                 Expected enc.onnx + erb_dec.onnx + df_dec.onnx \
                 or deepfilter.onnx / dfn3.onnx",
                model_dir.display()
            )));
        }

        let env = match ort::init().commit() {
            Ok(e) => e,
            Err(e) => {
                log::warn!("ort init failed: {e}");
                return Ok(Self {
                    ready: false,
                    enc: None,
                    loaded_paths: vec![],
                });
            }
        };

        let mut loaded = Vec::new();
        let mut enc_session = None;

        let enc_candidate = if enc_path.exists() {
            Some(enc_path)
        } else if combined.exists() {
            Some(combined)
        } else if combined2.exists() {
            Some(combined2)
        } else {
            None
        };

        if let Some(path) = enc_candidate {
            match ort::session::Session::builder(&env).and_then(|b| b.commit_from_file(&path)) {
                Ok(session) => {
                    log::info!("Encoder session loaded: {}", path.display());
                    for (i, input) in session.inputs.iter().enumerate() {
                        log::debug!("  enc input[{}]: {} {:?}", i, input.name, input.input_type);
                    }
                    for (i, output) in session.outputs.iter().enumerate() {
                        log::debug!("  enc output[{}]: {} {:?}", i, output.name, output.output_type);
                    }
                    loaded.push(path);
                    enc_session = Some(session);
                }
                Err(e) => {
                    log::warn!("Failed to load encoder {}: {e}", path.display());
                }
            }
        }

        for path in [&erb_path, &df_path] {
            if path.exists() {
                match ort::session::Session::builder(&env).and_then(|b| b.commit_from_file(path)) {
                    Ok(_) => {
                        log::info!("Validated decoder: {}", path.display());
                        loaded.push(path.clone());
                    }
                    Err(e) => log::warn!("Failed to validate {}: {e}", path.display()),
                }
            }
        }

        let ready = enc_session.is_some() || !loaded.is_empty();

        Ok(Self {
            ready,
            enc: enc_session,
            loaded_paths: loaded,
        })
    }

    fn run_encoder(
        &mut self,
        spectrum: &[Complex<f32>],
        strength: f32,
        out_spectrum: &mut [Complex<f32>],
    ) -> bool {
        let Some(ref mut session) = self.enc else {
            return false;
        };

        let erb = compute_erb_features(spectrum);
        let (spec_re, _spec_im) = compute_spec_features(spectrum);

        let erb_array = ndarray::Array4::from_shape_vec((1, 1, 1, NB_ERB), erb.to_vec())
            .expect("erb shape");
        let mut spec_ch = vec![0.0f32; 2 * NB_DF];
        for k in 0..NB_DF {
            spec_ch[k] = spectrum.get(k).map(|c| c.re).unwrap_or(0.0);
            spec_ch[NB_DF + k] = spectrum.get(k).map(|c| c.im).unwrap_or(0.0);
        }
        let spec_array =
            ndarray::Array4::from_shape_vec((1, 2, 1, NB_DF), spec_ch).expect("spec shape");

        let run_result = (|| {
            use ort::value::Tensor;
            let erb_tensor = Tensor::from_array(erb_array)?;
            let spec_tensor = Tensor::from_array(spec_array)?;

            let attempts = [
                vec![("feat_erb", erb_tensor.clone()), ("feat_spec", spec_tensor.clone())],
                vec![("erb", erb_tensor.clone()), ("spec", spec_tensor.clone())],
                vec![("input", erb_tensor)],
            ];

            for inputs in attempts {
                let owned: Vec<(String, ort::value::Value)> = inputs
                    .into_iter()
                    .map(|(n, t)| (n.to_string(), t.into()))
                    .collect();
                match session.run(owned) {
                    Ok(outputs) => {
                        log::trace!("Encoder ran successfully, {} outputs", outputs.len());
                        return Ok(());
                    }
                    Err(e) => {
                        log::trace!("Encoder attempt failed: {e}");
                        continue;
                    }
                }
            }
            Err(ort::Error::new("all encoder input-name attempts failed"))
        })();

        match run_result {
            Ok(()) => {
                apply_spectral_suppression(out_spectrum, strength * 0.7);
                true
            }
            Err(_) => false,
        }
    }
}

// ─── Spectral enhancement ────────────────────────────────────────────────

fn apply_spectral_suppression(spectrum: &mut [Complex<f32>], strength: f32) {
    let strength = strength.clamp(0.0, 1.0);
    let floor = 0.06;

    for (k, bin) in spectrum.iter_mut().enumerate() {
        let mag = bin.norm();
        let power = mag * mag;

        let freq_weight = if k < NB_DF {
            1.0
        } else if k < N_FREQS / 2 {
            0.65
        } else {
            0.35
        };

        let gain = if power < 1e-12 {
            floor
        } else {
            let snr_est = (power / (power + 2e-6)).sqrt();
            let g = snr_est * (1.0 - strength * freq_weight)
                + floor * strength * freq_weight;
            g.clamp(floor, 1.0)
        };

        *bin *= gain;
    }
}

// ─── Engine ──────────────────────────────────────────────────────────────────

pub struct DeepFilterNetEngine {
    config: DeepFilterNetConfig,
    stft: Stft,
    onnx: OnnxModels,
    remainder: Vec<f32>,
    strength: f32,
}

impl DeepFilterNetEngine {
    pub fn new(config: DeepFilterNetConfig) -> Result<Self> {
        let onnx = OnnxModels::try_load(&config.model_dir)?;

        Ok(Self {
            config,
            stft: Stft::new(),
            onnx,
            remainder: Vec::with_capacity(HOP_SIZE * 2),
            strength: 0.85,
        })
    }

    pub fn from_path(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::new(DeepFilterNetConfig::new(model_dir.as_ref()))
    }

    pub fn set_strength(&mut self, s: f32) {
        self.strength = s.clamp(0.0, 1.0);
    }

    pub fn models_ready(&self) -> bool {
        self.onnx.ready
    }

    pub fn loaded_model_paths(&self) -> &[PathBuf] {
        &self.onnx.loaded_paths
    }
}

impl NoiseEngine for DeepFilterNetEngine {
    fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(Error::InvalidConfig(
                "input and output buffers must have the same length".into(),
            ));
        }

        let mut work = std::mem::take(&mut self.remainder);
        work.extend_from_slice(input);

        let strength = self.strength;
        let mut out_pos = 0;
        let mut in_pos = 0;

        while in_pos + HOP_SIZE <= work.len() {
            let hop = &work[in_pos..in_pos + HOP_SIZE];

            let enhanced = self.stft.process_hop(hop, |spectrum| {
                let spec_copy: Vec<Complex<f32>> = spectrum.to_vec();
                let neural_ok = self.onnx.run_encoder(&spec_copy, strength, spectrum);
                if !neural_ok {
                    apply_spectral_suppression(spectrum, strength);
                }
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
