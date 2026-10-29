//! Minimal example of offline denoising.

use noise_cancellation::{Denoiser, DenoiserConfig, Strength};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = DenoiserConfig {
        strength: Strength(0.9),
        ..Default::default()
    };
    let mut den = Denoiser::new(config)?;

    // Fake 200 ms of audio
    let input = vec![0.01f32; 480 * 20];
    let mut output = vec![0.0f32; input.len()];

    den.process(&input, &mut output)?;
    println!(
        "Processed {} samples with {}",
        input.len(),
        den.engine_name()
    );
    Ok(())
}
