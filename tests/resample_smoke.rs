//! Resampling path smoke tests (RNNoise, no models needed).

#![cfg(feature = "rnnoise")]

use noise_cancellation::{Denoiser, DenoiserConfig, Strength, FRAME_SIZE};

#[test]
fn native_48k_unchanged_api() {
    let mut den = Denoiser::with_defaults().unwrap();
    assert_eq!(den.sample_rate(), 48_000);
    let n = FRAME_SIZE * 4;
    let input = vec![0.01f32; n];
    let mut output = vec![0.0f32; n];
    den.process(&input, &mut output).unwrap();
    assert!(output.iter().all(|x| x.is_finite()));
}

#[test]
fn rate_16k() {
    let mut den = Denoiser::with_sample_rate(16_000).unwrap();
    assert_eq!(den.sample_rate(), 16_000);
    let n = den.preferred_frame_size() * 8;
    let mut input = vec![0.0f32; n];
    for (i, s) in input.iter_mut().enumerate() {
        *s = 0.05 * (i as f32 * 0.1).sin();
    }
    let mut output = vec![0.0f32; n];
    den.process(&input, &mut output).unwrap();
    assert!(output.iter().all(|x| x.is_finite()));
    assert!(output.iter().any(|&x| x.abs() > 1e-8));
}

#[test]
fn rate_44100() {
    let mut den = Denoiser::new(DenoiserConfig {
        sample_rate: 44_100,
        strength: Strength(0.8),
        ..Default::default()
    })
    .unwrap();
    let n = den.preferred_frame_size() * 5;
    let input = vec![0.02f32; n];
    let mut output = vec![0.0f32; n];
    den.process(&input, &mut output).unwrap();
    assert!(output.iter().all(|x| x.is_finite()));
    den.reset();
    den.process(&input, &mut output).unwrap();
    assert!(output.iter().all(|x| x.is_finite()));
}

#[test]
fn strength_zero_pass_through_at_16k() {
    let mut den = Denoiser::new(DenoiserConfig {
        sample_rate: 16_000,
        strength: Strength(0.0),
        wet_dry_mix: true,
        ..Default::default()
    })
    .unwrap();
    let n = 320 * 4;
    let mut input = vec![0.0f32; n];
    for (i, s) in input.iter_mut().enumerate() {
        *s = 0.1 * (i as f32 / 16.0).sin();
    }
    let mut output = vec![0.0f32; n];
    den.process(&input, &mut output).unwrap();
    let max_diff = input
        .iter()
        .zip(output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-5, "max_diff={max_diff}");
}
