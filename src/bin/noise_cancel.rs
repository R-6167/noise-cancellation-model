//! Simple offline noise-cancellation CLI.

use clap::Parser;
use hound::{WavReader, WavWriter, SampleFormat};
use noise_cancellation::{Denoiser, DenoiserConfig, EngineKind, Strength, SAMPLE_RATE};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "noise-cancel", about = "High-quality noise cancellation (RNNoise / DeepFilterNet)")]
struct Args {
    /// Input WAV file (any common sample rate; will be resampled conceptually – currently expects 48 kHz)
    input: PathBuf,

    /// Output WAV file
    output: PathBuf,

    /// Suppression strength 0.0–1.0 (default 0.85)
    #[arg(short, long, default_value = "0.85")]
    strength: f32,

    /// Use maximum suppression (no dry mix)
    #[arg(long)]
    full: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut reader = WavReader::open(&args.input)?;
    let spec = reader.spec();

    if spec.channels != 1 {
        eprintln!("Warning: only mono is fully supported; using first channel");
    }

    // For v0.1 we require (or assume) 48 kHz. Real resampling can be added later.
    if spec.sample_rate != SAMPLE_RATE {
        eprintln!(
            "Warning: input is {} Hz, library expects {} Hz. Results may be suboptimal.",
            spec.sample_rate, SAMPLE_RATE
        );
    }

    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect(),
        SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
    };

    let config = DenoiserConfig {
        engine: EngineKind::Rnnoise,
        strength: Strength(args.strength),
        wet_dry_mix: !args.full,
    };

    let mut denoiser = Denoiser::new(config)?;
    println!("Using engine: {}", denoiser.engine_name());

    let mut output = vec![0.0f32; samples.len()];
    // Process in reasonably sized blocks for better cache behaviour
    const BLOCK: usize = 480 * 50; // 0.5 s blocks
    for (chunk_in, chunk_out) in samples.chunks(BLOCK).zip(output.chunks_mut(BLOCK)) {
        denoiser.process(chunk_in, chunk_out)?;
    }

    let out_spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(&args.output, out_spec)?;
    for &s in &output {
        let clamped = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(clamped)?;
    }
    writer.finalize()?;

    println!("Wrote cleaned audio to {}", args.output.display());
    Ok(())
}
