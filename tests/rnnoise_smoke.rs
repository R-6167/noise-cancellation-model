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
fn wet_dry_mix_aligns_delayed_dry_signal_with_wet_output() {
    let input: Vec<f32> = (0..FRAME_SIZE * 8)
        .map(|i| ((i * 13 % 97) as f32) / 100.0)
        .collect();
    let mut wet_only = Denoiser::new(DenoiserConfig {
        strength: Strength(1.0),
        wet_dry_mix: false,
        ..Default::default()
    }).unwrap();
    let mut mixed = Denoiser::new(DenoiserConfig {
        strength: Strength(0.5),
        wet_dry_mix: true,
        ..Default::default()
    }).unwrap();
    let latency = mixed.latency_samples();
    let mut wet = vec![0.0; input.len()];
    let mut output = vec![0.0; input.len()];
    wet_only.process(&input, &mut wet).unwrap();
    mixed.process(&input, &mut output).unwrap();

    for i in 0..output.len() {
        let dry = if i >= latency { input[i - latency] } else { 0.0 };
        let expected = dry * 0.5 + wet[i] * 0.5;
        assert!((output[i] - expected).abs() < 1e-6,
            "wet/dry alignment mismatch at {i}: {} vs {expected}", output[i]);
    }
}


#[test]
fn denoiser_flush_drains_latency_in_small_chunks_and_requires_reset() {
    let mut den = Denoiser::new(DenoiserConfig {
        wet_dry_mix: false,
        ..Default::default()
    }).unwrap();
    let input = vec![0.12f32; FRAME_SIZE * 3 + 101];
    let mut output = vec![0.0; input.len()];
    den.process(&input, &mut output).unwrap();

    let mut tail = Vec::new();
    loop {
        let mut chunk = vec![0.0; 73];
        let count = den.flush(&mut chunk).unwrap();
        tail.extend_from_slice(&chunk[..count]);
        if count == 0 { break; }
    }
    assert_eq!(tail.len(), den.latency_samples());
    assert!(tail.iter().all(|sample| sample.is_finite()));
    assert!(den.process(&input[..1], &mut [0.0]).is_err());

    den.reset();
    den.process(&input[..1], &mut [0.0]).unwrap();
}

#[test]
fn denoiser_flush_reports_host_rate_tail_length() {
    for sample_rate in [16_000, 44_100, 96_000] {
        let mut den = Denoiser::new(DenoiserConfig {
            sample_rate,
            wet_dry_mix: false,
            ..Default::default()
        }).unwrap();
        let input = vec![0.05; 1001];
        let mut output = vec![0.0; input.len()];
        den.process(&input, &mut output).unwrap();

        let mut tail = Vec::new();
        loop {
            let mut chunk = vec![0.0; 41];
            let count = den.flush(&mut chunk).unwrap();
            tail.extend_from_slice(&chunk[..count]);
            if count == 0 { break; }
        }
        assert_eq!(tail.len(), den.latency_samples(), "sample rate {sample_rate}");
        assert!(tail.iter().all(|sample| sample.is_finite()));
    }
}
