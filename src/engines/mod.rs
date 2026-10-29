//! Noise suppression engines.
//!
//! Both RNNoise and DeepFilterNet3 implement the same `NoiseEngine` trait
//! and are selected through the unified `Denoiser` API. From the caller's
//! perspective they ship as one algorithm family with different quality /
//! CPU trade-offs.

#[cfg(feature = "rnnoise")]
pub mod rnnoise;

#[cfg(feature = "deepfilternet")]
pub mod deepfilternet;

/// Common trait that every engine must implement.
///
/// This is the single contract that lets RNNoise and DeepFilterNet3
/// be interchangeable behind `Denoiser`.
pub trait NoiseEngine: Send {
    /// Process mono f32 audio at 48 kHz.
    /// Input and output lengths must match.
    fn process(&mut self, input: &[f32], output: &mut [f32]) -> crate::Result<()>;

    /// Reset internal state (call between unrelated streams).
    fn reset(&mut self);

    /// Human-readable name.
    fn name(&self) -> &'static str;

    /// Approximate algorithmic latency in samples (at 48 kHz).
    fn latency_samples(&self) -> usize;
}
