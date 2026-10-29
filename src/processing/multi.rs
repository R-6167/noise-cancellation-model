//! Multi-channel denoising: independent mono engines per channel.

use crate::processing::{Denoiser, DenoiserConfig};
use crate::{Error, Result};

/// Maximum channels supported (guards memory / CPU).
pub const MAX_CHANNELS: usize = 16;

/// Multi-channel denoiser: one independent mono engine state per channel.
///
/// Layout helpers:
/// - **Planar**: `process_planar` — slice-of-slices, one buffer per channel
/// - **Interleaved**: `process_interleaved` — LRLR… or multi-channel interleaved
pub struct MultiDenoiser {
    chans: Vec<Denoiser>,
}

impl MultiDenoiser {
    /// Create `channels` independent denoisers sharing the same config template.
    pub fn new(channels: usize, config: DenoiserConfig) -> Result<Self> {
        if channels == 0 || channels > MAX_CHANNELS {
            return Err(Error::InvalidConfig(format!(
                "channels must be 1..={MAX_CHANNELS}, got {channels}"
            )));
        }
        let mut chans = Vec::with_capacity(channels);
        for _ in 0..channels {
            chans.push(Denoiser::new(config.clone())?);
        }
        Ok(Self { chans })
    }

    /// Number of channels.
    pub fn channels(&self) -> usize {
        self.chans.len()
    }

    /// Host sample rate (same for all channels).
    pub fn sample_rate(&self) -> u32 {
        self.chans[0].sample_rate()
    }

    /// Preferred frame size at host rate (~10 ms).
    pub fn preferred_frame_size(&self) -> usize {
        self.chans[0].preferred_frame_size()
    }

    pub fn engine_name(&self) -> &'static str {
        self.chans[0].engine_name()
    }

    pub fn set_strength(&mut self, strength: crate::processing::Strength) {
        for c in &mut self.chans {
            c.set_strength(strength);
        }
    }

    pub fn reset(&mut self) {
        for c in &mut self.chans {
            c.reset();
        }
    }

    /// Process planar audio: `inputs[ch][frame]`, `outputs[ch][frame]`.
    pub fn process_planar(&mut self, inputs: &[&[f32]], outputs: &mut [&mut [f32]]) -> Result<()> {
        if inputs.len() != self.chans.len() || outputs.len() != self.chans.len() {
            return Err(Error::InvalidConfig(format!(
                "expected {} channels, got in={} out={}",
                self.chans.len(),
                inputs.len(),
                outputs.len()
            )));
        }
        let n = inputs.first().map(|s| s.len()).unwrap_or(0);
        for (i, (inp, out)) in inputs.iter().zip(outputs.iter_mut()).enumerate() {
            if inp.len() != n || out.len() != n {
                return Err(Error::InvalidConfig(format!(
                    "channel {i}: length mismatch (expected {n} frames)"
                )));
            }
            self.chans[i].process(inp, out)?;
        }
        Ok(())
    }

    /// Process interleaved audio: `frames` frames × `channels` samples.
    pub fn process_interleaved(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        frames: usize,
    ) -> Result<()> {
        let ch = self.chans.len();
        let need = frames.checked_mul(ch).ok_or_else(|| {
            Error::InvalidConfig("frames * channels overflow".into())
        })?;
        if input.len() < need || output.len() < need {
            return Err(Error::InvalidConfig(format!(
                "buffer too small: need {need}, got in={} out={}",
                input.len(),
                output.len()
            )));
        }
        if frames == 0 {
            return Ok(());
        }

        let mut planar_in: Vec<Vec<f32>> = (0..ch).map(|_| vec![0.0; frames]).collect();
        let mut planar_out: Vec<Vec<f32>> = (0..ch).map(|_| vec![0.0; frames]).collect();

        for f in 0..frames {
            for c in 0..ch {
                planar_in[c][f] = input[f * ch + c];
            }
        }

        for c in 0..ch {
            self.chans[c].process(&planar_in[c], &mut planar_out[c])?;
        }

        for f in 0..frames {
            for c in 0..ch {
                output[f * ch + c] = planar_out[c][f];
            }
        }
        Ok(())
    }
}
