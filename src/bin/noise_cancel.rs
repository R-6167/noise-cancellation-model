//! Simple offline noise-cancellation CLI.

use clap::Parser;
use hound::{SampleFormat, WavReader, WavWriter};
use noise_cancellation::{Denoiser, DenoiserConfig, EngineKind, Strength};
use std::{env, fs, path::PathBuf};

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

    /// Engine to use: "rnnoise" or "deepfilternet" (overrides the saved preference)
    #[arg(long)]
    engine: Option<String>,

    /// Save the selected engine as the default for future runs
    #[arg(long)]
    remember_engine: bool,

    /// Path to DeepFilterNet3 model directory (required when --engine deepfilternet)
    #[arg(long)]
    model_dir: Option<PathBuf>,
}

fn engine_preference_path() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        env::var_os("APPDATA").map(PathBuf::from).map(|path| {
            path.join("noise-cancellation-model").join("engine")
        })
    }
    #[cfg(target_os = "macos")]
    {
        env::var_os("HOME").map(PathBuf::from).map(|path| {
            path.join("Library")
                .join("Application Support")
                .join("noise-cancellation-model")
                .join("engine")
        })
    }
    #[cfg(all(not(windows), not(target_os = "macos")))]
    {
        env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
            .map(|path| path.join("noise-cancellation-model").join("engine"))
    }
}

fn load_engine_preference() -> Option<String> {
    let preference = fs::read_to_string(engine_preference_path()?).ok()?;
    let engine = preference.trim();
    if engine.is_empty() {
        None
    } else {
        Some(engine.to_owned())
    }
}

fn save_engine_preference(engine: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = engine_preference_path()
        .ok_or("could not determine the user config directory for engine preference")?;
    let parent = path
        .parent()
        .ok_or("engine preference path has no parent directory")?;
    fs::create_dir_all(parent)?;
    fs::write(path, format!("{engine}\n"))?;
    Ok(())
}

fn first_channel(samples: Vec<f32>, channels: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if channels == 0 || !samples.len().is_multiple_of(channels) {
        return Err("WAV sample count is not a whole number of frames".into());
    }
    if channels == 1 {
        return Ok(samples);
    }
    Ok(samples
        .chunks_exact(channels)
        .map(|frame| frame[0])
        .collect())
}

fn process_offline(
    denoiser: &mut Denoiser,
    samples: &[f32],
    block_size: usize,
    dry_bypass: bool,
) -> noise_cancellation::Result<Vec<f32>> {
    let block_size = block_size.max(1);
    let mut processed = vec![0.0f32; samples.len()];
    for (chunk_in, chunk_out) in samples
        .chunks(block_size)
        .zip(processed.chunks_mut(block_size))
    {
        denoiser.process(chunk_in, chunk_out)?;
    }

    let latency = denoiser.latency_samples();
    let mut tail = Vec::new();
    loop {
        let mut buffer = vec![0.0f32; latency.max(1).min(4096)];
        let written = denoiser.flush(&mut buffer)?;
        tail.extend_from_slice(&buffer[..written]);
        if written == 0 {
            break;
        }
    }

    // Strength zero with wet/dry mixing is a true immediate pass-through; trimming
    // algorithmic latency in that case would remove valid input samples.
    if dry_bypass {
        return Ok(processed);
    }

    let mut stream = processed;
    stream.extend(tail);
    let start = latency.min(stream.len());
    let end = start.saturating_add(samples.len()).min(stream.len());
    let mut duration_matched = stream[start..end].to_vec();
    duration_matched.resize(samples.len(), 0.0);
    Ok(duration_matched)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut reader = WavReader::open(&args.input)?;
    let spec = reader.spec();

    let raw_samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Int => reader
            .samples::<i16>()
            .map(|sample| sample.map(|value| value as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
        SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
    };
    let samples = first_channel(raw_samples, spec.channels as usize)?;

    if spec.sample_rate != 48_000 {
        eprintln!(
            "Resampling internally from {} Hz and writing at the original rate.",
            spec.sample_rate
        );
    }
    if spec.channels > 1 {
        eprintln!(
            "Input has {} channels; processing the first channel and writing mono output.",
            spec.channels
        );
    }

    let selected_engine = args
        .engine
        .clone()
        .or_else(load_engine_preference)
        .unwrap_or_else(|| "rnnoise".to_owned());

    let engine = match selected_engine.as_str() {
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
        sample_rate: spec.sample_rate,
    };

    let mut denoiser = Denoiser::new(config)?;
    if args.remember_engine {
        save_engine_preference(&selected_engine)?;
        println!("Saved preferred engine: {selected_engine}");
    }
    println!("Using engine: {}", denoiser.engine_name());

    let block_size = (spec.sample_rate as usize / 2).max(1);
    let dry_bypass = !args.full && args.strength.clamp(0.0, 1.0) == 0.0;
    let output = process_offline(&mut denoiser, &samples, block_size, dry_bypass)?;

    let out_spec = hound::WavSpec {
        channels: 1,
        sample_rate: spec.sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(&args.output, out_spec)?;
    for &sample in &output {
        let clamped = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(clamped)?;
    }
    writer.finalize()?;

    println!(
        "Wrote {} samples of cleaned audio to {}",
        output.len(),
        args.output.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{first_channel, process_offline};
    use noise_cancellation::{Denoiser, DenoiserConfig, Strength, FRAME_SIZE};

    #[test]
    fn offline_processing_preserves_duration_after_tail_flush() {
        let mut denoiser = Denoiser::new(DenoiserConfig {
            strength: Strength(1.0),
            wet_dry_mix: false,
            ..Default::default()
        })
        .unwrap();
        let input: Vec<f32> = (0..FRAME_SIZE * 3 + 113)
            .map(|i| (i as f32 * 0.03).sin() * 0.1)
            .collect();
        let output = process_offline(&mut denoiser, &input, 127, false).unwrap();
        assert_eq!(output.len(), input.len());
        assert!(output.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn zero_strength_dry_bypass_is_bit_exact() {
        let input: Vec<f32> = (0..FRAME_SIZE + 17)
            .map(|i| (i as f32 * 0.02).sin() * 0.1)
            .collect();
        let mut denoiser = Denoiser::new(DenoiserConfig {
            strength: Strength(0.0),
            wet_dry_mix: true,
            ..Default::default()
        })
        .unwrap();
        let output = process_offline(&mut denoiser, &input, 127, true).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn multichannel_input_uses_first_channel_and_preserves_frame_count() {
        let input = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        assert_eq!(first_channel(input, 2).unwrap(), vec![1.0, 2.0, 3.0]);
    }
}
