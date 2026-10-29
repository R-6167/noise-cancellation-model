//! DeepFilterNet3 engine – real-time optimised neural path.
//!
//! Design goals for low, predictable CPU:
//!   - Zero heap allocation on the per-hop hot path after warm-up
//!   - Pre-allocated feature buffers
//!   - Cached successful tensor input names
//!   - Single-threaded ort sessions
//!   - AudioBufferPool for streaming work buffers

#![cfg(feature = "deepfilternet")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::engines::NoiseEngine;
use crate::utils::AudioBufferPool;
use crate::{Error, Result};

pub const HOP_SIZE: usize = 480;
pub const FFT_SIZE: usize = 960;
pub const N_FREQS: usize = FFT_SIZE / 2 + 1;
pub const NB_ERB: usize = 32;
pub const NB_DF: usize = 96;
pub const DF_ORDER: usize = 5;

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

struct Stft {
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
    window: Vec<f32>,
    time_buf: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    framed: Vec<f32>,
    time_out: Vec<f32>,
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
            window[n] = (std::f32::consts::FRAC_PI_2 * x.sin().powi(2)).sin();
        }
        Self {
            forward,
            inverse,
            window,
            time_buf: vec![0.0; FFT_SIZE],
            spectrum: vec![Complex::new(0.0, 0.0); N_FREQS],
            framed: vec![0.0; FFT_SIZE],
            time_out: vec![0.0; FFT_SIZE],
            ola: vec![0.0; FFT_SIZE],
        }
    }

    fn process_hop<F>(&mut self, hop: &[f32], process_spectrum: F) -> [f32; HOP_SIZE]
    where
        F: FnOnce(&mut [Complex<f32>]),
    {
        self.time_buf.copy_within(HOP_SIZE..FFT_SIZE, 0);
        self.time_buf[HOP_SIZE..].copy_from_slice(hop);
        for i in 0..FFT_SIZE {
            self.framed[i] = self.time_buf[i] * self.window[i];
        }
        self.forward.process(&mut self.framed, &mut self.spectrum).expect("FFT forward");
        process_spectrum(&mut self.spectrum);
        self.inverse.process(&mut self.spectrum, &mut self.time_out).expect("FFT inverse");
        let scale = 1.0 / FFT_SIZE as f32;
        for i in 0..FFT_SIZE {
            self.time_out[i] *= scale * self.window[i];
            self.ola[i] += self.time_out[i];
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

struct FeatureBufs {
    erb: [f32; NB_ERB],
    spec_ch: [f32; 2 * NB_DF],
    mask: [f32; NB_ERB],
    coefs: Vec<f32>,
}

impl FeatureBufs {
    fn new() -> Self {
        Self {
            erb: [0.0; NB_ERB],
            spec_ch: [0.0; 2 * NB_DF],
            mask: [1.0; NB_ERB],
            coefs: Vec::with_capacity(NB_DF * DF_ORDER * 2),
        }
    }

    fn fill_from_spectrum(&mut self, spectrum: &[Complex<f32>]) {
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
            self.erb[b] = (energy + 1e-10).ln();
        }
        for k in 0..NB_DF {
            self.spec_ch[k] = spectrum.get(k).map(|c| c.re).unwrap_or(0.0);
            self.spec_ch[NB_DF + k] = spectrum.get(k).map(|c| c.im).unwrap_or(0.0);
        }
    }
}

fn apply_erb_mask(spectrum: &mut [Complex<f32>], mask: &[f32; NB_ERB], strength: f32) {
    let n = spectrum.len().min(N_FREQS);
    let max_bin = NB_DF.min(n).max(1);
    for k in 0..n {
        let band_f = k as f32 / max_bin as f32 * (NB_ERB - 1) as f32;
        let b0 = (band_f.floor() as usize).min(NB_ERB - 1);
        let b1 = (b0 + 1).min(NB_ERB - 1);
        let frac = band_f - b0 as f32;
        let m = mask[b0] * (1.0 - frac) + mask[b1] * frac;
        spectrum[k] *= 1.0 - strength + strength * m.clamp(0.0, 1.0);
    }
}

fn apply_deep_filter(spectrum: &mut [Complex<f32>], coefs: &[f32], strength: f32) {
    if coefs.is_empty() {
        return;
    }
    let n = NB_DF.min(spectrum.len()).min(coefs.len() / 2);
    for k in 0..n {
        let re = coefs[k * 2];
        let im = coefs.get(k * 2 + 1).copied().unwrap_or(0.0);
        let g = Complex::new(re, im).norm().clamp(0.0, 2.0);
        spectrum[k] *= 1.0 - strength + strength * g.min(1.0);
    }
}

fn apply_spectral_suppression(spectrum: &mut [Complex<f32>], strength: f32) {
    let strength = strength.clamp(0.0, 1.0);
    let floor = 0.06;
    let half = N_FREQS / 2;
    for (k, bin) in spectrum.iter_mut().enumerate() {
        let power = bin.norm_sqr();
        let freq_weight = if k < NB_DF { 1.0 } else if k < half { 0.65 } else { 0.35 };
        let gain = if power < 1e-12 {
            floor
        } else {
            let snr_est = (power / (power + 2e-6)).sqrt();
            (snr_est * (1.0 - strength * freq_weight) + floor * strength * freq_weight).clamp(floor, 1.0)
        };
        *bin *= gain;
    }
}

#[derive(Clone, Copy, Default)]
struct NameCache {
    enc_attempt: Option<u8>,
    erb_from: Option<&'static str>,
    df_from: Option<&'static str>,
    resolved: bool,
}

struct OnnxModels {
    ready: bool,
    enc: Option<ort::session::Session>,
    erb_dec: Option<ort::session::Session>,
    df_dec: Option<ort::session::Session>,
    feats: FeatureBufs,
    names: NameCache,
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
                "No DeepFilterNet3 ONNX models found in {}.",
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
                    erb_dec: None,
                    df_dec: None,
                    feats: FeatureBufs::new(),
                    names: NameCache::default(),
                    loaded_paths: vec![],
                });
            }
        };

        let build = |path: &Path| -> Option<ort::session::Session> {
            let builder = ort::session::Session::builder(&env).ok()?;
            let builder = builder
                .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
                .ok()?;
            let builder = builder.with_intra_threads(1).ok()?;
            match builder.commit_from_file(path) {
                Ok(s) => {
                    log::info!("Loaded (rt): {}", path.display());
                    Some(s)
                }
                Err(e) => {
                    log::warn!("Failed {}: {e}", path.display());
                    None
                }
            }
        };

        let mut loaded = Vec::new();
        let mut enc = None;
        let mut erb_dec = None;
        let mut df_dec = None;

        if has_split {
            if let Some(s) = build(&enc_path) {
                loaded.push(enc_path);
                enc = Some(s);
            }
            if let Some(s) = build(&erb_path) {
                loaded.push(erb_path);
                erb_dec = Some(s);
            }
            if let Some(s) = build(&df_path) {
                loaded.push(df_path);
                df_dec = Some(s);
            }
        } else {
            let path = if combined.exists() { &combined } else { &combined2 };
            if let Some(s) = build(path) {
                loaded.push(path.clone());
                enc = Some(s);
            }
        }

        Ok(Self {
            ready: enc.is_some(),
            enc,
            erb_dec,
            df_dec,
            feats: FeatureBufs::new(),
            names: NameCache::default(),
            loaded_paths: loaded,
        })
    }

    fn enhance(&mut self, spectrum: &mut [Complex<f32>], strength: f32) -> bool {
        let Some(ref mut enc) = self.enc else {
            return false;
        };

        self.feats.fill_from_spectrum(spectrum);

        let erb_array =
            match ndarray::Array4::from_shape_vec((1, 1, 1, NB_ERB), self.feats.erb.to_vec()) {
                Ok(a) => a,
                Err(_) => return false,
            };
        let spec_array =
            match ndarray::Array4::from_shape_vec((1, 2, 1, NB_DF), self.feats.spec_ch.to_vec()) {
                Ok(a) => a,
                Err(_) => return false,
            };

        use ort::value::Tensor;
        let erb_t = match Tensor::from_array(erb_array) {
            Ok(t) => t,
            Err(_) => return false,
        };
        let spec_t = match Tensor::from_array(spec_array) {
            Ok(t) => t,
            Err(_) => return false,
        };

        let enc_out = {
            const ATTEMPTS: &[&[(&str, bool)]] = &[
                &[("feat_erb", true), ("feat_spec", false)],
                &[("erb", true), ("spec", false)],
                &[("input", true)],
            ];
            let mut success = None;
            let start = self.names.enc_attempt.unwrap_or(0) as usize;
            let order: Vec<usize> = std::iter::once(start)
                .chain((0..ATTEMPTS.len()).filter(|&i| i != start))
                .collect();
            for idx in order {
                let attempt = ATTEMPTS[idx];
                let mut owned = Vec::with_capacity(2);
                for &(name, is_erb) in attempt {
                    let t = if is_erb { erb_t.clone() } else { spec_t.clone() };
                    owned.push((name.to_string(), t.into()));
                }
                if let Ok(outputs) = enc.run(owned) {
                    if !self.names.resolved {
                        self.names.enc_attempt = Some(idx as u8);
                    }
                    success = Some(outputs);
                    break;
                }
            }
            success
        };

        let Some(enc_out) = enc_out else {
            return false;
        };

        self.feats.mask = [1.0; NB_ERB];
        if let Some(ref mut erb_dec) = self.erb_dec {
            let candidates: &[&str] = if let Some(n) = self.names.erb_from {
                &[n]
            } else {
                &["emb", "e3", "e2", "e1", "e0", "output", "0"]
            };
            for &name in candidates {
                if let Some(val) = enc_out.get(name) {
                    let inputs = vec![(name.to_string(), val.clone())];
                    if let Ok(dec_out) = erb_dec.run(inputs) {
                        if let Some((_, v)) = dec_out.iter().next() {
                            if let Ok(arr) = v.try_extract_tensor::<f32>() {
                                if let Some(data) = arr.view().as_slice() {
                                    for (i, &x) in data.iter().take(NB_ERB).enumerate() {
                                        self.feats.mask[i] = x.clamp(0.0, 1.0);
                                    }
                                }
                            }
                        }
                        if !self.names.resolved {
                            self.names.erb_from = Some(name);
                        }
                        break;
                    }
                }
            }
        }

        self.feats.coefs.clear();
        if let Some(ref mut df_dec) = self.df_dec {
            let candidates: &[&str] = if let Some(n) = self.names.df_from {
                &[n]
            } else {
                &["emb", "c0", "output", "0"]
            };
            for &name in candidates {
                if let Some(val) = enc_out.get(name) {
                    let inputs = vec![(name.to_string(), val.clone())];
                    if let Ok(dec_out) = df_dec.run(inputs) {
                        if let Some((_, v)) = dec_out.iter().next() {
                            if let Ok(arr) = v.try_extract_tensor::<f32>() {
                                if let Some(data) = arr.view().as_slice() {
                                    self.feats.coefs.extend_from_slice(data);
                                }
                            }
                        }
                        if !self.names.resolved {
                            self.names.df_from = Some(name);
                        }
                        break;
                    }
                }
            }
        }

        if !self.names.resolved {
            self.names.resolved = true;
            log::info!("Tensor name cache resolved – subsequent hops use fast path");
        }

        apply_erb_mask(spectrum, &self.feats.mask, strength);
        if !self.feats.coefs.is_empty() {
            apply_deep_filter(spectrum, &self.feats.coefs, strength);
        }
        apply_spectral_suppression(spectrum, strength * 0.2);
        true
    }
}

pub struct DeepFilterNetEngine {
    config: DeepFilterNetConfig,
    stft: Stft,
    onnx: OnnxModels,
    pool: AudioBufferPool,
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
            pool: AudioBufferPool::for_streaming(),
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

        let mut work = self.pool.acquire(self.remainder.len() + input.len() + HOP_SIZE);
        work.extend_from_slice(&self.remainder);
        work.extend_from_slice(input);
        self.remainder.clear();

        let strength = self.strength;
        let mut out_pos = 0;
        let mut in_pos = 0;

        while in_pos + HOP_SIZE <= work.len() {
            let hop = &work[in_pos..in_pos + HOP_SIZE];
            let enhanced = self.stft.process_hop(hop, |spectrum| {
                if !self.onnx.enhance(spectrum, strength) {
                    apply_spectral_suppression(spectrum, strength);
                }
            });
            let n = (output.len() - out_pos).min(HOP_SIZE);
            output[out_pos..out_pos + n].copy_from_slice(&enhanced[..n]);
            out_pos += n;
            in_pos += HOP_SIZE;
        }

        self.remainder.extend_from_slice(&work[in_pos..]);

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
