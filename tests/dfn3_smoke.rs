//! DeepFilterNet3 smoke tests against vendored models in `models/`.
//!
//! Run with:
//!   cargo test --features deepfilternet --test dfn3_smoke
//!
//! Fails clearly if ONNX models are missing from models/.

#![cfg(feature = "deepfilternet")]

use noise_cancellation::{
    find_model_dir, is_model_dir, Denoiser, DenoiserConfig, EngineKind, Strength, FRAME_SIZE,
    SAMPLE_RATE,
};
use std::path::PathBuf;

fn model_dir() -> Option<PathBuf> {
    find_model_dir().or_else(|| {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
        if is_model_dir(&p) {
            Some(p)
        } else {
            None
        }
    })
}

fn require_models() -> PathBuf {
    model_dir().expect(
        "DeepFilterNet3 models not found. Expected models/enc.onnx, erb_dec.onnx, df_dec.onnx",
    )
}

#[test]
fn models_are_vendored() {
    let dir = require_models();
    assert!(dir.join("enc.onnx").is_file());
    assert!(dir.join("erb_dec.onnx").is_file());
    assert!(dir.join("df_dec.onnx").is_file());
}

#[test]
fn creates_engine_and_reports_ready() {
    let dir = require_models();
    let den = Denoiser::with_deepfilternet(&dir).expect("create DFN3 denoiser");
    assert_eq!(den.engine_name(), "DeepFilterNet3");
    assert_eq!(den.sample_rate(), SAMPLE_RATE);
    assert!(den.latency_samples() > 0);
    assert_eq!(den.preferred_frame_size(), FRAME_SIZE);

    let eng = noise_cancellation::engines::deepfilternet::DeepFilterNetEngine::from_path(&dir)
        .expect("from_path");
    assert!(
        eng.models_ready(),
        "ONNX sessions should load from vendored models"
    );
    assert!(
        !eng.loaded_model_paths().is_empty(),
        "at least one model path should be recorded"
    );
}

#[test]
fn processes_silence_without_nan() {
    let dir = require_models();
    let mut den = Denoiser::with_deepfilternet(&dir).unwrap();
    let n = FRAME_SIZE * 20;
    let input = vec![0.0f32; n];
    let mut output = vec![1.0f32; n];
    den.process(&input, &mut output).expect("process silence");
    assert!(
        output.iter().all(|x| x.is_finite()),
        "output must be finite"
    );
    let energy: f32 = output.iter().map(|x| x * x).sum();
    assert!(energy < 1.0, "silence energy unexpectedly high: {energy}");
}

#[test]
fn processes_noise_and_tone() {
    let dir = require_models();
    let mut den = Denoiser::new(DenoiserConfig {
        engine: EngineKind::DeepFilterNet { model_dir: dir },
        strength: Strength(0.9),
        wet_dry_mix: false,
        sample_rate: SAMPLE_RATE,
    })
    .unwrap();

    let n = FRAME_SIZE * 30;
    let mut input = vec![0.0f32; n];
    for (i, s) in input.iter_mut().enumerate() {
        let t = i as f32 / SAMPLE_RATE as f32;
        *s = 0.15 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            + 0.03 * (((i * 31 + 7) % 200) as f32 / 200.0 - 0.5);
    }

    let mut output = vec![0.0f32; n];
    den.process(&input, &mut output).expect("process tone+noise");

    assert!(output.iter().all(|x| x.is_finite()));
    assert!(
        output.iter().any(|&x| x.abs() > 1e-6),
        "output should not be all zeros"
    );
    let peak = output.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(peak < 5.0, "peak too high: {peak}");
}

#[test]
fn reset_then_process() {
    let dir = require_models();
    let mut den = Denoiser::with_deepfilternet(&dir).unwrap();
    let input = vec![0.02f32; FRAME_SIZE * 4];
    let mut output = vec![0.0f32; input.len()];
    den.process(&input, &mut output).unwrap();
    den.reset();
    den.process(&input, &mut output).unwrap();
    assert!(output.iter().all(|x| x.is_finite()));
}

#[test]
fn strength_controls_wet_dry() {
    let dir = require_models();
    let n = FRAME_SIZE * 10;
    let mut input = vec![0.0f32; n];
    for (i, s) in input.iter_mut().enumerate() {
        let t = i as f32 / SAMPLE_RATE as f32;
        *s = 0.1 * (2.0 * std::f32::consts::PI * 200.0 * t).sin();
    }

    let mut dry = Denoiser::new(DenoiserConfig {
        engine: EngineKind::DeepFilterNet {
            model_dir: dir.clone(),
        },
        strength: Strength(0.0),
        wet_dry_mix: true,
        sample_rate: SAMPLE_RATE,
    })
    .unwrap();
    let mut out_dry = vec![0.0f32; n];
    dry.process(&input, &mut out_dry).unwrap();

    let max_diff = input
        .iter()
        .zip(out_dry.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 0.05,
        "strength=0 should nearly pass through, max_diff={max_diff}"
    );
}

#[test]
fn auto_discover_models() {
    if model_dir().is_none() {
        eprintln!("skip auto_discover: no models");
        return;
    }
    let den = Denoiser::with_deepfilternet_auto();
    assert!(den.is_ok(), "auto discover failed: {:?}", den.err());
    assert_eq!(den.unwrap().engine_name(), "DeepFilterNet3");
}

#[test]
fn streaming_chunks_match_block_energy_order() {
    let dir = require_models();
    let n = FRAME_SIZE * 16;
    let mut input = vec![0.0f32; n];
    for (i, s) in input.iter_mut().enumerate() {
        *s = 0.05 * ((i % 97) as f32 / 97.0);
    }

    let mut den1 = Denoiser::with_deepfilternet(&dir).unwrap();
    let mut out1 = vec![0.0f32; n];
    den1.process(&input, &mut out1).unwrap();

    let mut den2 = Denoiser::with_deepfilternet(&dir).unwrap();
    let mut out2 = vec![0.0f32; n];
    let chunk = 127;
    for start in (0..n).step_by(chunk) {
        let end = (start + chunk).min(n);
        den2.process(&input[start..end], &mut out2[start..end])
            .unwrap();
    }

    assert!(out1.iter().all(|x| x.is_finite()));
    assert!(out2.iter().all(|x| x.is_finite()));
    let max_diff = out1.iter().zip(&out2)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff <= 1e-6, "chunk boundaries changed output; max_diff={max_diff}");
    let e1: f32 = out1.iter().map(|x| x * x).sum();
    let e2: f32 = out2.iter().map(|x| x * x).sum();
    assert!(e1.is_finite() && e2.is_finite());
}
