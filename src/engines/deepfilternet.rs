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
