//! RNNoise engine via the pure-Rust `nnnoiseless` crate.
//! Extremely lightweight, excellent for real-time and low-power devices.

use crate::engines::NoiseEngine;
use crate::utils::AudioBufferPool;
use crate::{Error, Result, FRAME_SIZE};
use nnnoiseless::DenoiseState;
use std::collections::VecDeque;

/// RNNoise-based suppressor.
pub struct RnnoiseEngine {
    state: Box<DenoiseState<'static>>,
    frame_in: [f32; FRAME_SIZE],
    frame_out: [f32; FRAME_SIZE],
    remainder: Vec<f32>,
    pool: AudioBufferPool,
    output_queue: VecDeque<f32>,
    flushing: bool,
    flush_remaining: usize,
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
            // Keep the documented frame latency constant regardless of caller block size.
            output_queue: std::iter::repeat_n(0.0, FRAME_SIZE).collect(),
            flushing: false,
            flush_remaining: 0,
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
        if self.flushing {
            return Err(Error::InvalidConfig("stream has been flushed; call reset before processing more audio".into()));
        }
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

        let mut in_pos = 0;
        while in_pos + FRAME_SIZE <= work.len() {
            for i in 0..FRAME_SIZE {
                self.frame_in[i] = work[in_pos + i] * 32768.0;
            }

            self.state.process_frame(&mut self.frame_out, &self.frame_in);

            self.output_queue
                .extend(self.frame_out.iter().map(|sample| sample / 32768.0));
            in_pos += FRAME_SIZE;
        }

        self.remainder.extend_from_slice(&work[in_pos..]);

        for sample in output.iter_mut() {
            *sample = self.output_queue.pop_front().unwrap_or(0.0);
        }

        Ok(())
    }

    fn flush(&mut self, output: &mut [f32]) -> Result<usize> {
        if output.is_empty() || self.flush_remaining == 0 && self.flushing {
            return Ok(0);
        }
        if !self.flushing {
            self.flushing = true;
            self.flush_remaining = self.latency_samples();
            if !self.remainder.is_empty() {
                self.frame_in.fill(0.0);
                self.frame_in[..self.remainder.len()].copy_from_slice(&self.remainder);
                for sample in self.frame_in.iter_mut() {
                    *sample *= 32768.0;
                }
                self.state.process_frame(&mut self.frame_out, &self.frame_in);
                self.output_queue.extend(self.frame_out.iter().map(|sample| sample / 32768.0));
                self.remainder.clear();
            }
        }
        let count = output.len().min(self.flush_remaining);
        for sample in output.iter_mut().take(count) {
            *sample = self.output_queue.pop_front().unwrap_or(0.0);
        }
        self.flush_remaining -= count;
        if self.flush_remaining == 0 {
            self.output_queue.clear();
        }
        Ok(count)
    }

    fn reset(&mut self) {
        self.state = DenoiseState::new();
        self.remainder.clear();
        self.output_queue.clear();
        self.output_queue.extend(std::iter::repeat_n(0.0, FRAME_SIZE));
        self.flushing = false;
        self.flush_remaining = 0;
    }

    fn name(&self) -> &'static str {
        "RNNoise"
    }

    fn latency_samples(&self) -> usize {
        FRAME_SIZE
    }
}


#[cfg(test)]
mod tests {
    use super::RnnoiseEngine;
    use crate::engines::NoiseEngine;

    fn render_in_chunks(input: &[f32], chunk_size: usize) -> Vec<f32> {
        let mut engine = RnnoiseEngine::new();
        let mut rendered = vec![0.0; input.len()];
        let mut offset = 0;
        while offset < input.len() {
            let end = (offset + chunk_size).min(input.len());
            engine.process(&input[offset..end], &mut rendered[offset..end]).unwrap();
            offset = end;
        }
        rendered
    }

    #[test]
    fn output_is_independent_of_input_chunk_boundaries() {
        let input: Vec<f32> = (0..9600)
            .map(|i| (i as f32 * 0.037).sin() * 0.2)
            .collect();
        let whole = render_in_chunks(&input, input.len());
        for chunk_size in [1, 37, 127, 480, 1024, 2048] {
            let chunked = render_in_chunks(&input, chunk_size);
            assert_eq!(whole.len(), chunked.len());
            for (i, (a, b)) in whole.iter().zip(&chunked).enumerate() {
                assert!((a - b).abs() <= 1e-6, "mismatch at {i} for chunk {chunk_size}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn reset_clears_partial_input_and_reestablishes_latency() {
        let mut engine = RnnoiseEngine::new();
        let input = vec![0.2; 100];
        let mut output = vec![1.0; input.len()];
        engine.process(&input, &mut output).unwrap();
        engine.reset();
        output.fill(1.0);
        engine.process(&input, &mut output).unwrap();
        assert!(output.iter().all(|sample| *sample == 0.0));
    }

    fn render_with_tail(input: &[f32], chunk_size: usize, tail_chunk_size: usize) -> Vec<f32> {
        let mut engine = RnnoiseEngine::new();
        let mut rendered = vec![0.0; input.len()];
        let mut offset = 0;
        while offset < input.len() {
            let end = (offset + chunk_size).min(input.len());
            engine.process(&input[offset..end], &mut rendered[offset..end]).unwrap();
            offset = end;
        }
        let mut tail = Vec::new();
        loop {
            let mut buf = vec![0.0; tail_chunk_size];
            let count = engine.flush(&mut buf).unwrap();
            tail.extend_from_slice(&buf[..count]);
            if count == 0 {
                break;
            }
        }
        assert_eq!(tail.len(), engine.latency_samples());
        rendered.extend(tail);
        rendered
    }

    #[test]
    fn flush_preserves_final_partial_frame_and_is_chunk_invariant() {
        for input_len in [1, 100, 479, 480, 481, 959, 960, 961, 4097] {
            let input: Vec<f32> = (0..input_len)
                .map(|i| (i as f32 * 0.031).sin() * 0.2)
                .collect();
            let reference = render_with_tail(&input, input_len.max(1), FRAME_SIZE);
            for chunk in [1, 37, 127, 480, 1024] {
                let actual = render_with_tail(&input, chunk, 73);
                assert_eq!(reference.len(), actual.len(), "length mismatch for input {input_len}");
                for (i, (a, b)) in reference.iter().zip(&actual).enumerate() {
                    assert!((a - b).abs() <= 1e-6,
                        "flush output mismatch at {i}, input {input_len}, chunk {chunk}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn flush_requires_reset_before_more_input_and_reset_reopens_stream() {
        let mut engine = RnnoiseEngine::new();
        let input = vec![0.1; 100];
        let mut output = vec![0.0; input.len()];
        engine.process(&input, &mut output).unwrap();
        let mut tail = vec![0.0; 31];
        assert_eq!(engine.flush(&mut tail).unwrap(), 31);
        assert!(engine.process(&input, &mut output).is_err());
        engine.reset();
        engine.process(&input, &mut output).unwrap();
    }

}
