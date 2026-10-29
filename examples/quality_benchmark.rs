//! Compare RNNoise and DeepFilterNet on the same noisy speech with a clean reference.
//!
//! Usage:
//! cargo run --release --features cli --example quality_benchmark -- clean.wav noisy.wav [model_dir]
//!
//! Inputs must be aligned mono/first-channel WAVs at 48 kHz. The benchmark removes
//! the engine's reported startup latency after flushing its final tail.

use hound::{SampleFormat, WavReader};
use noise_cancellation::{
    Denoiser, DenoiserConfig, EngineKind, Strength, SAMPLE_RATE,
};
use std::error::Error;
use std::path::{Path, PathBuf};

fn read_first_channel(path: &Path) -> Result<(u32, Vec<f32>), Box<dyn Error>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channels == 0 {
        return Err(format!("{} has zero channels", path.display()).into());
    }

    let raw: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        SampleFormat::Int if spec.bits_per_sample <= 16 => reader
            .samples::<i16>()
            .map(|sample| sample.map(|value| value as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
        SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<Result<Vec<_>, _>>()?
        }
    };

    if !raw.len().is_multiple_of(channels) {
        return Err(format!("{} has an incomplete final audio frame", path.display()).into());
    }
    let mono = if channels == 1 {
        raw
    } else {
        raw.chunks_exact(channels).map(|frame| frame[0]).collect()
    };
    Ok((spec.sample_rate, mono))
}

fn process_full_stream(
    engine: EngineKind,
    noisy: &[f32],
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut denoiser = Denoiser::new(DenoiserConfig {
        engine,
        strength: Strength(0.85),
        wet_dry_mix: false,
        sample_rate: SAMPLE_RATE,
    })?;

    let mut processed = vec![0.0f32; noisy.len()];
    const BLOCK: usize = 480 * 50;
    for (input, output) in noisy.chunks(BLOCK).zip(processed.chunks_mut(BLOCK)) {
        denoiser.process(input, output)?;
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

    processed.extend(tail);
    let end = latency
        .checked_add(noisy.len())
        .ok_or("audio length overflow while trimming latency")?;
    if end > processed.len() {
        return Err("flushed output did not contain enough samples to restore duration".into());
    }
    Ok(processed[latency..end].to_vec())
}

fn snr_db(clean: &[f32], estimate: &[f32]) -> Option<f64> {
    if clean.is_empty() || clean.len() != estimate.len() {
        return None;
    }
    let signal_power: f64 = clean.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let error_power: f64 = clean
        .iter()
        .zip(estimate)
        .map(|(&reference, &actual)| {
            let error = actual as f64 - reference as f64;
            error * error
        })
        .sum();
    if signal_power <= 1e-20 {
        return None;
    }
    if error_power <= 1e-20 {
        return Some(f64::INFINITY);
    }
    Some(10.0 * (signal_power / error_power).log10())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        return Err(
            "usage: quality_benchmark <clean.wav> <noisy.wav> [model_dir]".into(),
        );
    }
    let model_dir = PathBuf::from(args.get(3).map(String::as_str).unwrap_or("models"));
    let (clean_rate, clean) = read_first_channel(Path::new(&args[1]))?;
    let (noisy_rate, noisy) = read_first_channel(Path::new(&args[2]))?;
    if clean_rate != noisy_rate {
        return Err("clean and noisy WAVs must have the same sample rate".into());
    }
    if clean_rate != SAMPLE_RATE {
        return Err(format!(
            "quality benchmark requires aligned 48 kHz inputs; got {} Hz. Convert both files to 48 kHz first.",
            clean_rate
        ).into());
    }
    if clean.len() != noisy.len() {
        return Err(format!(
            "clean/noisy duration mismatch: clean={} samples, noisy={} samples",
            clean.len(), noisy.len()
        ).into());
    }

    let input_snr = snr_db(&clean, &noisy).ok_or("cannot calculate input SNR")?;
    println!("Input SNR: {input_snr:.2} dB");
    println!("Strength: 0.85, wet/dry mix disabled, sample rate: {SAMPLE_RATE} Hz");

    let rnnoise = process_full_stream(EngineKind::Rnnoise, &noisy)?;
    let rn_snr = snr_db(&clean, &rnnoise).ok_or("cannot calculate RNNoise SNR")?;
    println!("RNNoise: {rn_snr:.2} dB SNR ({:+.2} dB vs noisy input)", rn_snr - input_snr);

    let deepfilter = process_full_stream(
        EngineKind::DeepFilterNet { model_dir: model_dir.clone() },
        &noisy,
    )?;
    let df_snr = snr_db(&clean, &deepfilter).ok_or("cannot calculate DeepFilterNet SNR")?;
    println!(
        "DeepFilterNet3: {df_snr:.2} dB SNR ({:+.2} dB vs noisy input)",
        df_snr - input_snr
    );
    println!("These scores apply only to this aligned reference pair; listen for speech artifacts as well.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::snr_db;

    #[test]
    fn snr_metric_rewards_an_exact_clean_signal() {
        let clean = [0.2f32, -0.1, 0.4, 0.3];
        let noisy = [0.3f32, -0.2, 0.3, 0.5];
        assert!(snr_db(&clean, &clean).unwrap().is_infinite());
        assert!(snr_db(&clean, &clean).unwrap() > snr_db(&clean, &noisy).unwrap());
    }

    #[test]
    fn snr_rejects_mismatched_lengths_and_silent_reference() {
        assert!(snr_db(&[1.0], &[1.0, 2.0]).is_none());
        assert!(snr_db(&[0.0, 0.0], &[0.1, 0.2]).is_none());
    }
}
