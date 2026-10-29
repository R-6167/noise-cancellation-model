//! Noise suppression engines.

#[cfg(feature = "rnnoise")]
pub mod rnnoise;

#[cfg(feature = "deepfilternet")]
pub mod deepfilternet;

/// Common trait that every engine must implement.
pub trait NoiseEngine: Send {
    /// Process one or more frames of mono f32 audio at 48 kHz.
    /// Input and output lengths must match.
    fn process(&mut self, input: &[f32], output: &mut [f32]) -> crate::Result<()>;

    /// Reset internal state (call between unrelated streams).
    fn reset(&mut self);

    /// Human-readable name.
    fn name(&self) -> &'static str;

    /// Approximate algorithmic latency in samples (at 48 kHz).
    fn latency_samples(&self) -> usize;
}
