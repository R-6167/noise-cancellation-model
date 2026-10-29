//! DeepFilterNet3 – true DSP path matching the official contract.
//!
//! See README for pipeline. Official ERB widths, EMA + unit norm,
//! order-5 deep-filter FIR with spectrum history.

#![cfg(feature = "deepfilternet")]

use std::collections::VecDeque;
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
pub const DF_LOOKAHEAD: usize = 2;

const ERB_WIDTHS: [usize; NB_ERB] = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7, 7, 8, 10, 12, 13, 15, 18, 20,
    24, 28, 31, 37, 42, 50, 56, 67,
];
const NORM_ALPHA: f32 = 0.99;

#[derive(Debug, Clone)]
pub struct DeepFilterNetConfig {
    pub model_dir: PathBuf,
    pub atten_lim_db: Option<f32>,
    pub compensate_delay: bool,
}

impl DeepFilterNetConfig {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self { model_dir: model_dir.into(), atten_lim_db: None, compensate_delay: true }
    }
}

struct ErbBank {
    fwd: [[f32; N_FREQS]; NB_ERB],
    inv: [[f32; NB_ERB]; N_FREQS],
}

impl ErbBank {
    fn new() -> Self {
        let mut fwd = [[0.0f32; N_FREQS]; NB_ERB];
        let mut inv = [[0.0f32; NB_ERB]; N_FREQS];
        let mut bin = 0usize;
        for (b, &w) in ERB_WIDTHS.iter().enumerate() {
            let scale = 1.0 / w as f32;
            for k in 0..w {
                if bin + k < N_FREQS {
                    fwd[b][bin + k] = scale;
                    inv[bin + k][b] = 1.0;
                }
            }
            bin += w;
        }
        Self { fwd, inv }
    }

    fn analyze(&self, power: &[f32]) -> [f32; NB_ERB] {
        let mut out = [0.0f32; NB_ERB];
        for b in 0..NB_ERB {
            let mut e = 0.0f32;
            for k in 0..N_FREQS { e += power[k] * self.fwd[b][k]; }
            out[b] = e;
        }
        out
    }

    fn synthesize_mask(&self, mask: &[f32; NB_ERB]) -> [f32; N_FREQS] {
        let mut g = [0.0f32; N_FREQS];
        for k in 0..N_FREQS {
            let mut v = 0.0f32;
            for b in 0..NB_ERB { v += mask[b] * self.inv[k][b]; }
            g[k] = v.clamp(0.0, 1.0);
        }
        g
    }
}

struct ErbNorm { mean: [f32; NB_ERB], alpha: f32 }
impl ErbNorm {
    fn new() -> Self {
        let mut mean = [0.0f32; NB_ERB];
        for b in 0..NB_ERB { mean[b] = -60.0 - 30.0 * (b as f32 / (NB_ERB - 1) as f32); }
        Self { mean, alpha: NORM_ALPHA }
    }
    fn normalize(&mut self, erb_db: &mut [f32; NB_ERB]) {
        for b in 0..NB_ERB {
            self.mean[b] = self.alpha * self.mean[b] + (1.0 - self.alpha) * erb_db[b];
            erb_db[b] -= self.mean[b];
        }
    }
    fn reset(&mut self) { *self = Self::new(); }
}

struct SpecNorm { state: [f32; NB_DF], alpha: f32 }
impl SpecNorm {
    fn new() -> Self {
        let mut state = [0.0f32; NB_DF];
        for k in 0..NB_DF { state[k] = 0.001 - 0.0009 * (k as f32 / (NB_DF - 1) as f32); }
        Self { state, alpha: NORM_ALPHA }
    }
    fn normalize(&mut self, re: &mut [f32; NB_DF], im: &mut [f32; NB_DF]) {
        for k in 0..NB_DF {
            let e = re[k] * re[k] + im[k] * im[k];
            self.state[k] = self.alpha * self.state[k] + (1.0 - self.alpha) * e;
            let s = (self.state[k] + 1e-12).sqrt();
            re[k] /= s; im[k] /= s;
        }
    }
    fn reset(&mut self) { *self = Self::new(); }
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
            forward, inverse, window,
            time_buf: vec![0.0; FFT_SIZE],
            spectrum: vec![Complex::new(0.0, 0.0); N_FREQS],
            framed: vec![0.0; FFT_SIZE],
            time_out: vec![0.0; FFT_SIZE],
            ola: vec![0.0; FFT_SIZE],
        }
    }

    fn process_hop<F>(&mut self, hop: &[f32], process_spectrum: F) -> [f32; HOP_SIZE]
    where F: FnOnce(&mut [Complex<f32>]) {
        self.time_buf.copy_within(HOP_SIZE..FFT_SIZE, 0);
        self.time_buf[HOP_SIZE..].copy_from_slice(hop);
        for i in 0..FFT_SIZE { self.framed[i] = self.time_buf[i] * self.window[i]; }
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

struct DfHistory { frames: VecDeque<[Complex<f32>; NB_DF]> }
impl DfHistory {
    fn new() -> Self {
        let mut frames = VecDeque::with_capacity(DF_ORDER + DF_LOOKAHEAD);
        for _ in 0..DF_ORDER { frames.push_back([Complex::new(0.0, 0.0); NB_DF]); }
        Self { frames }
    }
    fn push(&mut self, spectrum: &[Complex<f32>]) {
        let mut frame = [Complex::new(0.0, 0.0); NB_DF];
        for k in 0..NB_DF { frame[k] = spectrum.get(k).copied().unwrap_or(Complex::new(0.0, 0.0)); }
        if self.frames.len() >= DF_ORDER { self.frames.pop_front(); }
        self.frames.push_back(frame);
    }
    fn apply(&self, coefs: &[f32], out: &mut [Complex<f32>]) {
        let n_frames = self.frames.len().min(DF_ORDER);
        for k in 0..NB_DF.min(out.len()) {
            let mut acc = Complex::new(0.0, 0.0);
            for n in 0..n_frames {
                let c_re = coefs.get((n * NB_DF + k) * 2).copied().unwrap_or(0.0);
                let c_im = coefs.get((n * NB_DF + k) * 2 + 1).copied().unwrap_or(0.0);
                let idx = self.frames.len() - n_frames + n;
                acc += Complex::new(c_re, c_im) * self.frames[idx][k];
            }
            out[k] = acc;
        }
    }
    fn reset(&mut self) { *self = Self::new(); }
}

struct OnnxModels {
    ready: bool,
    combined: Option<ort::session::Session>,
    enc: Option<ort::session::Session>,
    erb_dec: Option<ort::session::Session>,
    df_dec: Option<ort::session::Session>,
    loaded_paths: Vec<PathBuf>,
}

impl OnnxModels {
    fn try_load(model_dir: &Path) -> Result<Self> {
        let combined_path = model_dir.join("deepfilter.onnx");
        let combined2 = model_dir.join("dfn3.onnx");
        let enc_path = model_dir.join("enc.onnx");
        let erb_path = model_dir.join("erb_dec.onnx");
        let df_path = model_dir.join("df_dec.onnx");
        let has_combined = combined_path.exists() || combined2.exists();
        let has_split = enc_path.exists() && erb_path.exists() && df_path.exists();
        if !has_combined && !has_split {
            return Err(Error::Engine(format!("No DeepFilterNet3 ONNX in {}", model_dir.display())));
        }
        let env = match ort::init().commit() {
            Ok(e) => e,
            Err(e) => {
                log::warn!("ort init failed: {e}");
                return Ok(Self { ready: false, combined: None, enc: None, erb_dec: None, df_dec: None, loaded_paths: vec![] });
            }
        };
        let build = |path: &Path| -> Option<ort::session::Session> {
            let b = ort::session::Session::builder(&env).ok()?;
            let b = b.with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3).ok()?;
            let b = b.with_intra_threads(1).ok()?;
            match b.commit_from_file(path) {
                Ok(s) => { log::info!("DFN3 loaded: {}", path.display()); Some(s) }
                Err(e) => { log::warn!("Failed {}: {e}", path.display()); None }
            }
        };
        let mut loaded = Vec::new();
        let mut combined = None;
        let mut enc = None;
        let mut erb_dec = None;
        let mut df_dec = None;
        if has_combined {
            let p = if combined_path.exists() { &combined_path } else { &combined2 };
            if let Some(s) = build(p) { loaded.push(p.clone()); combined = Some(s); }
        }
        if has_split {
            if let Some(s) = build(&enc_path) { loaded.push(enc_path); enc = Some(s); }
            if let Some(s) = build(&erb_path) { loaded.push(erb_path); erb_dec = Some(s); }
            if let Some(s) = build(&df_path) { loaded.push(df_path); df_dec = Some(s); }
        }
        Ok(Self { ready: combined.is_some() || enc.is_some(), combined, enc, erb_dec, df_dec, loaded_paths: loaded })
    }
}

pub struct DeepFilterNetEngine {
    config: DeepFilterNetConfig,
    stft: Stft,
    erb: ErbBank,
    erb_norm: ErbNorm,
    spec_norm: SpecNorm,
    df_hist: DfHistory,
    onnx: OnnxModels,
    pool: AudioBufferPool,
    remainder: Vec<f32>,
    strength: f32,
    power: [f32; N_FREQS],
    erb_feat: [f32; NB_ERB],
    spec_re: [f32; NB_DF],
    spec_im: [f32; NB_DF],
    mask: [f32; NB_ERB],
    lin_gain: [f32; N_FREQS],
    df_coefs: Vec<f32>,
}

impl DeepFilterNetEngine {
    pub fn new(config: DeepFilterNetConfig) -> Result<Self> {
        let onnx = OnnxModels::try_load(&config.model_dir)?;
        Ok(Self {
            config, stft: Stft::new(), erb: ErbBank::new(), erb_norm: ErbNorm::new(),
            spec_norm: SpecNorm::new(), df_hist: DfHistory::new(), onnx,
            pool: AudioBufferPool::for_streaming(), remainder: Vec::with_capacity(HOP_SIZE * 2),
            strength: 0.85, power: [0.0; N_FREQS], erb_feat: [0.0; NB_ERB],
            spec_re: [0.0; NB_DF], spec_im: [0.0; NB_DF], mask: [1.0; NB_ERB],
            lin_gain: [1.0; N_FREQS], df_coefs: Vec::with_capacity(DF_ORDER * NB_DF * 2),
        })
    }

    pub fn from_path(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::new(DeepFilterNetConfig::new(model_dir.as_ref()))
    }

    pub fn set_strength(&mut self, s: f32) { self.strength = s.clamp(0.0, 1.0); }
    pub fn models_ready(&self) -> bool { self.onnx.ready }
    pub fn loaded_model_paths(&self) -> &[PathBuf] { &self.onnx.loaded_paths }

    fn enhance_spectrum(&mut self, spectrum: &mut [Complex<f32>]) {
        let strength = self.strength;
        for k in 0..N_FREQS.min(spectrum.len()) { self.power[k] = spectrum[k].norm_sqr(); }
        let erb_e = self.erb.analyze(&self.power);
        for b in 0..NB_ERB { self.erb_feat[b] = 10.0 * (erb_e[b] + 1e-10).log10(); }
        self.erb_norm.normalize(&mut self.erb_feat);
        for k in 0..NB_DF {
            self.spec_re[k] = spectrum.get(k).map(|c| c.re).unwrap_or(0.0);
            self.spec_im[k] = spectrum.get(k).map(|c| c.im).unwrap_or(0.0);
        }
        self.spec_norm.normalize(&mut self.spec_re, &mut self.spec_im);
        self.df_hist.push(spectrum);
        self.mask = [1.0; NB_ERB];
        self.df_coefs.clear();
        let neural_ok = self.run_neural();
        if neural_ok { self.lin_gain = self.erb.synthesize_mask(&self.mask); }
        else { self.lin_gain = [1.0; N_FREQS]; }
        if let Some(lim_db) = self.config.atten_lim_db {
            let floor = 10f32.powf(-lim_db.abs() / 20.0);
            for g in self.lin_gain.iter_mut() { *g = g.max(floor); }
        }
        let mut df_out = [Complex::new(0.0, 0.0); NB_DF];
        if neural_ok && !self.df_coefs.is_empty() { self.df_hist.apply(&self.df_coefs, &mut df_out); }
        for k in 0..spectrum.len().min(N_FREQS) {
            if k < NB_DF && neural_ok && !self.df_coefs.is_empty() {
                spectrum[k] = spectrum[k] * (1.0 - strength) + df_out[k] * strength;
            } else {
                spectrum[k] *= 1.0 - strength + strength * self.lin_gain[k];
            }
        }
    }

    fn run_neural(&mut self) -> bool {
        let erb_array = match ndarray::Array4::from_shape_vec((1, 1, 1, NB_ERB), self.erb_feat.to_vec()) {
            Ok(a) => a, Err(_) => return false,
        };
        let mut spec_ch = vec![0.0f32; 2 * NB_DF];
        for k in 0..NB_DF { spec_ch[k] = self.spec_re[k]; spec_ch[NB_DF + k] = self.spec_im[k]; }
        let spec_array = match ndarray::Array4::from_shape_vec((1, 2, 1, NB_DF), spec_ch) {
            Ok(a) => a, Err(_) => return false,
        };
        use ort::value::Tensor;
        let erb_t = match Tensor::from_array(erb_array) { Ok(t) => t, Err(_) => return false };
        let spec_t = match Tensor::from_array(spec_array) { Ok(t) => t, Err(_) => return false };

        if let Some(ref mut session) = self.onnx.combined {
            let inputs = vec![("feat_erb".to_string(), erb_t.clone().into()), ("feat_spec".to_string(), spec_t.clone().into())];
            if let Ok(outputs) = session.run(inputs) { self.extract_outputs(&outputs); return true; }
        }

        let Some(ref mut enc) = self.onnx.enc else { return false; };
        let inputs = vec![("feat_erb".to_string(), erb_t.into()), ("feat_spec".to_string(), spec_t.into())];
        let Ok(enc_out) = enc.run(inputs) else { return false; };

        if let Some(ref mut erb_dec) = self.onnx.erb_dec {
            let mut dec_in = Vec::new();
            for n in ["emb", "e3", "e2", "e1", "e0"] {
                if let Some(v) = enc_out.get(n) { dec_in.push((n.to_string(), v.clone())); }
            }
            if !dec_in.is_empty() {
                if let Ok(out) = erb_dec.run(dec_in) {
                    if let Some((_, v)) = out.iter().next() {
                        if let Ok(arr) = v.try_extract_tensor::<f32>() {
                            if let Some(data) = arr.view().as_slice() {
                                for (i, &x) in data.iter().take(NB_ERB).enumerate() { self.mask[i] = x.clamp(0.0, 1.0); }
                            }
                        }
                    }
                }
            }
        }

        if let Some(ref mut df_dec) = self.onnx.df_dec {
            let mut dec_in = Vec::new();
            for n in ["emb", "c0"] {
                if let Some(v) = enc_out.get(n) { dec_in.push((n.to_string(), v.clone())); }
            }
            if !dec_in.is_empty() {
                if let Ok(out) = df_dec.run(dec_in) {
                    if let Some((_, v)) = out.iter().next() {
                        if let Ok(arr) = v.try_extract_tensor::<f32>() {
                            if let Some(data) = arr.view().as_slice() { self.df_coefs.extend_from_slice(data); }
                        }
                    }
                }
            }
        }
        true
    }

    fn extract_outputs(&mut self, outputs: &ort::session::SessionOutputs) {
        if let Some(v) = outputs.get("erb_mask").or_else(|| outputs.get("m")) {
            if let Ok(arr) = v.try_extract_tensor::<f32>() {
                if let Some(data) = arr.view().as_slice() {
                    for (i, &x) in data.iter().take(NB_ERB).enumerate() { self.mask[i] = x.clamp(0.0, 1.0); }
                }
            }
        }
        if let Some(v) = outputs.get("df_coefs").or_else(|| outputs.get("coefs")) {
            if let Ok(arr) = v.try_extract_tensor::<f32>() {
                if let Some(data) = arr.view().as_slice() { self.df_coefs.extend_from_slice(data); }
            }
        }
    }
}

impl NoiseEngine for DeepFilterNetEngine {
    fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(Error::InvalidConfig("input and output buffers must have the same length".into()));
        }
        let mut work = self.pool.acquire(self.remainder.len() + input.len() + HOP_SIZE);
        work.extend_from_slice(&self.remainder);
        work.extend_from_slice(input);
        self.remainder.clear();
        let mut out_pos = 0;
        let mut in_pos = 0;
        while in_pos + HOP_SIZE <= work.len() {
            let hop = &work[in_pos..in_pos + HOP_SIZE];
            let enhanced = self.stft.process_hop(hop, |spectrum| { self.enhance_spectrum(spectrum); });
            let n = (output.len() - out_pos).min(HOP_SIZE);
            output[out_pos..out_pos + n].copy_from_slice(&enhanced[..n]);
            out_pos += n;
            in_pos += HOP_SIZE;
        }
        self.remainder.extend_from_slice(&work[in_pos..]);
        while out_pos < output.len() { output[out_pos] = 0.0; out_pos += 1; }
        Ok(())
    }

    fn reset(&mut self) {
        self.remainder.clear();
        self.stft.reset();
        self.erb_norm.reset();
        self.spec_norm.reset();
        self.df_hist.reset();
    }

    fn name(&self) -> &'static str { "DeepFilterNet3" }

    fn latency_samples(&self) -> usize {
        if self.config.compensate_delay { HOP_SIZE * (1 + DF_LOOKAHEAD + 1) }
        else { HOP_SIZE * (1 + DF_LOOKAHEAD) }
    }
}
