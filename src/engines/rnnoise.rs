//! RNNoise engine via the pure-Rust `nnnoiseless` crate.
//! Extremely lightweight, excellent for real-time and low-power devices.

use crate::engines::NoiseEngine;
use crate::utils::AudioBufferPool;
use crate::{Error, Result, FRAME_SIZE};
use nnnoiseless::DenoiseState;

/// RNNoise-based suppressor.
pub struct RnnoiseEngine {
    state: Box<DenoiseState<'static>>,
    frame_in: [f32; FRAME_SIZE],
    frame_out: [f32; FRAME_SIZE],
    remainder: Vec<f32>,
    pool: AudioBufferPool,
}

impl RnnoiseEngine {
    /// Create a new RNNoise engine with the built-in model.
    pub fn new() -> Self {
        Self {
            state: DenoiseState::new(),
            frame_in: [0.0; FRAME_SIZE],
            frame_out: [0.0; FRAME_SIZE],
            remainder: Vec::with_capacity(FRAME_SIZE),
            pool: AudioBufferPool::for_streaming(),
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

        let mut work = self
            .pool
            .acquire(self.remainder.len() + input.len() + FRAME_SIZE);
        work.extend_from_slice(&self.remainder);
        work.extend_from_slice(input);
        self.remainder.clear();

        let mut out_pos = 0;
        let mut in_pos = 0;

        while in_pos + FRAME_SIZE <= work.len() {
            for i in 0..FRAME_SIZE {
                self.frame_in[i] = work[in_pos + i] * 32768.0;
            }

            self.state
                .process_frame(&mut self.frame_out, &self.frame_in);

            for i in 0..FRAME_SIZE {
                if out_pos < output.len() {
                    output[out_pos] = self.frame_out[i] / 32768.0;
                    out_pos += 1;
                }
            }
            in_pos += FRAME_SIZE;
        }

        self.remainder.extend_from_slice(&work[in_pos..]);

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
        FRAME_SIZE
    }
}
