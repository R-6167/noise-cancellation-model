//! RNNoise engine via the pure-Rust `nnnoiseless` crate.
//! Extremely lightweight, excellent for real-time and low-power devices.

use nnnoiseless::DenoiseState;
use crate::engines::NoiseEngine;
use crate::{Error, Result, FRAME_SIZE};

/// RNNoise-based suppressor.
pub struct RnnoiseEngine {
    state: Box<DenoiseState<'static>>,
    /// Temporary buffer for the 480-sample frames RNNoise expects.
    frame_in: [f32; FRAME_SIZE],
    frame_out: [f32; FRAME_SIZE],
    /// Carry-over samples when input length is not a multiple of FRAME_SIZE.
    remainder: Vec<f32>,
}

impl RnnoiseEngine {
    /// Create a new RNNoise engine with the built-in model.
    pub fn new() -> Self {
        Self {
            state: DenoiseState::new(),
            frame_in: [0.0; FRAME_SIZE],
            frame_out: [0.0; FRAME_SIZE],
            remainder: Vec::with_capacity(FRAME_SIZE),
        }
    }
}

impl Default for RnnoiseEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl NoiseEngine for RnnoiseEngine {
    fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(Error::InvalidConfig(
                "input and output buffers must have the same length".into(),
            ));
        }

        // Append new input to any remainder from previous call
        let mut work = std::mem::take(&mut self.remainder);
        work.extend_from_slice(input);

        let mut out_pos = 0;
        let mut in_pos = 0;

        while in_pos + FRAME_SIZE <= work.len() {
            // RNNoise expects samples roughly in i16 range (±32768)
            for i in 0..FRAME_SIZE {
                self.frame_in[i] = work[in_pos + i] * 32768.0;
            }

            self.state.process_frame(&mut self.frame_out, &self.frame_in);

            for i in 0..FRAME_SIZE {
                if out_pos < output.len() {
                    output[out_pos] = self.frame_out[i] / 32768.0;
                    out_pos += 1;
                }
            }
            in_pos += FRAME_SIZE;
        }

        // Save leftover samples for next call
        self.remainder = work[in_pos..].to_vec();

        // If we produced fewer samples than requested (first frames), zero-fill
        // the rest so the caller always gets a full buffer.
        while out_pos < output.len() {
            output[out_pos] = 0.0;
            out_pos += 1;
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.state = DenoiseState::new();
        self.remainder.clear();
    }

    fn name(&self) -> &'static str {
        "RNNoise"
    }

    fn latency_samples(&self) -> usize {
        // RNNoise is essentially one frame of delay in streaming use.
        FRAME_SIZE
    }
}
