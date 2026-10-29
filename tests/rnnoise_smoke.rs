//! Self-contained smoke test – no model files, no network.

use noise_cancellation::{Denoiser, DenoiserConfig, Strength, FRAME_SIZE};

#[test]
fn rnnoise_processes_silence() {
    let mut den = Denoiser::with_defaults().expect("RNNoise must work offline");
    let input = vec![0.0f32; FRAME_SIZE * 5];
    let mut output = vec![0.0f32; input.len()];
    den.process(&input, &mut output).unwrap();
    let energy: f32 = output.iter().map(|x| x * x).sum();
    assert!(energy < 1e-4, "energy={energy}");
}

#[test]
fn rnnoise_processes_noise() {
    let mut den = Denoiser::new(DenoiserConfig {
        strength: Strength(1.0),
        wet_dry_mix: false,
        ..Default::default()
    })
    .unwrap();

    let mut input = vec![0.0f32; FRAME_SIZE * 10];
    for (i, s) in input.iter_mut().enumerate() {
        let t = i as f32 / 48000.0;
        *s = 0.1 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            + 0.02 * ((i * 17) % 100) as f32 / 100.0;
    }
    let mut output = vec![0.0f32; input.len()];
    den.process(&input, &mut output).unwrap();
    assert_eq!(output.len(), input.len());
    assert!(output.iter().any(|&x| x.abs() > 1e-6));
}

#[test]
fn reset_is_safe() {
    let mut den = Denoiser::with_defaults().unwrap();
    den.reset();
    let input = vec![0.01f32; FRAME_SIZE];
    let mut output = vec![0.0f32; FRAME_SIZE];
    den.process(&input, &mut output).unwrap();
}


#[test]
fn wet_dry_mix_keeps_dry_signal_aligned_to_reported_latency() {
    let mut den = Denoiser::new(DenoiserConfig {
        strength: Strength(0.0),
        wet_dry_mix: true,
        ..Default::default()
    }).unwrap();
    let latency = den.latency_samples();
    let input: Vec<f32> = (0..FRAME_SIZE * 6)
        .map(|i| ((i * 13 % 97) as f32) / 100.0)
        .collect();
    let mut output = vec![0.0; input.len()];
    den.process(&input, &mut output).unwrap();

    assert!(output[..latency].iter().all(|sample| sample.abs() < 1e-7));
    for i in latency..output.len() {
        assert!((output[i] - input[i - latency]).abs() < 1e-7,
            "dry signal misaligned at {i}: {} vs {}", output[i], input[i - latency]);
    }
}
