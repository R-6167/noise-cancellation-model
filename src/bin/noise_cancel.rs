//! Simple offline noise-cancellation CLI.

use clap::Parser;
use hound::{SampleFormat, WavReader, WavWriter};
use noise_cancellation::{Denoiser, DenoiserConfig, EngineKind, Strength, SAMPLE_RATE};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "noise-cancel",
    about = "High-quality noise cancellation (RNNoise / DeepFilterNet3)"
)]
struct Args {
    /// Input WAV file
    input: PathBuf,

    /// Output WAV file
    output: PathBuf,

    /// Suppression strength 0.0–1.0 (default 0.85)
    #[arg(short, long, default_value = "0.85")]
    strength: f32,

    /// Use maximum suppression (no dry mix)
    #[arg(long)]
    full: bool,

    /// Engine to use: "rnnoise" (default) or "deepfilternet"
    #[arg(long, default_value = "rnnoise")]
    engine: String,

    /// Path to DeepFilterNet3 model directory (required when --engine deepfilternet)
    #[arg(long)]
    model_dir: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut reader = WavReader::open(&args.input)?;
    let spec = reader.spec();

    if spec.channels != 1 {
        eprintln!("Warning: only mono is fully supported; using first channel");
    }

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

    let engine = match args.engine.as_str() {
        "rnnoise" => {
            #[cfg(feature = "rnnoise")]
            {
                EngineKind::Rnnoise
            }
            #[cfg(not(feature = "rnnoise"))]
            {
                return Err("rnnoise feature not enabled".into());
            }
        }
        "deepfilternet" | "dfn3" | "df" => {
            #[cfg(feature = "deepfilternet")]
            {
                let model_dir = args.model_dir.unwrap_or_else(|| PathBuf::from("models"));
                EngineKind::DeepFilterNet { model_dir }
            }
            #[cfg(not(feature = "deepfilternet"))]
            {
                return Err(
                    "deepfilternet feature not enabled – rebuild with --features deepfilternet"
                        .into(),
                );
            }
        }
        other => {
            return Err(format!(
                "unknown engine '{}', use 'rnnoise' or 'deepfilternet'",
                other
            )
            .into());
        }
    };

    let config = DenoiserConfig {
        engine,
        strength: Strength(args.strength),
        wet_dry_mix: !args.full,
        sample_rate: SAMPLE_RATE,
    };

    let mut denoiser = Denoiser::new(config)?;
    println!("Using engine: {}", denoiser.engine_name());

    let mut output = vec![0.0f32; samples.len()];
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
