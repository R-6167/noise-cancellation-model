//! Multi-channel smoke tests (RNNoise).

#![cfg(feature = "rnnoise")]

use noise_cancellation::{DenoiserConfig, MultiDenoiser, Strength, MAX_CHANNELS};

#[test]
fn stereo_interleaved() {
    let mut m = MultiDenoiser::new(2, DenoiserConfig::default()).unwrap();
    assert_eq!(m.channels(), 2);
    let frames = m.preferred_frame_size() * 4;
    let mut input = vec![0.0f32; frames * 2];
    for f in 0..frames {
        input[f * 2] = 0.05 * (f as f32 * 0.07).sin();
        input[f * 2 + 1] = 0.04 * (f as f32 * 0.05).cos();
    }
    let mut output = vec![0.0f32; input.len()];
    m.process_interleaved(&input, &mut output, frames).unwrap();
    assert!(output.iter().all(|x| x.is_finite()));
}

#[test]
fn planar_stereo() {
    let mut m = MultiDenoiser::new(2, DenoiserConfig::default()).unwrap();
    let n = m.preferred_frame_size() * 3;
    let left_in = vec![0.02f32; n];
    let right_in = vec![0.03f32; n];
    let mut left_out = vec![0.0f32; n];
    let mut right_out = vec![0.0f32; n];
    let inputs: [&[f32]; 2] = [&left_in, &right_in];
    let mut outs: [&mut [f32]; 2] = [&mut left_out, &mut right_out];
    m.process_planar(&inputs, &mut outs).unwrap();
    assert!(left_out.iter().all(|x| x.is_finite()));
    assert!(right_out.iter().all(|x| x.is_finite()));
}

#[test]
fn rejects_zero_and_too_many() {
    assert!(MultiDenoiser::new(0, DenoiserConfig::default()).is_err());
    assert!(MultiDenoiser::new(MAX_CHANNELS + 1, DenoiserConfig::default()).is_err());
}

#[test]
fn strength_and_reset() {
    let mut m = MultiDenoiser::new(2, DenoiserConfig::default()).unwrap();
    m.set_strength(Strength(0.5));
    m.reset();
    let frames = 480;
    let input = vec![0.01f32; frames * 2];
    let mut output = vec![0.0f32; frames * 2];
    m.process_interleaved(&input, &mut output, frames).unwrap();
}
