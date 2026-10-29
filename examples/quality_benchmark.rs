//! Compare RNNoise and DeepFilterNet on the same noisy speech with a clean reference.
//!
//! Usage:
//! cargo run --release --features cli --example quality_benchmark -- clean.wav noisy.wav [model_dir]
//!
//! Inputs must be aligned mono/first-channel WAVs at 48 kHz. Reported SNR is
//! calculated both at zero lag and with a bounded diagnostic lag search; the latter
//! is diagnostic only and must not hide alignment or latency bugs.

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
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

fn write_mono_wav(path: &Path, samples: &[f32]) -> Result<(), Box<dyn Error>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for &sample in samples {
        let value = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(value)?;
    }
    writer.finalize()?;
    Ok(())
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

/// Return the overlapping regions for a lag. Positive lag means the estimate
/// starts later than the clean reference; negative lag means it starts earlier.
fn overlap_for_lag<'a>(clean: &'a [f32], estimate: &'a [f32], lag: isize) -> (&'a [f32], &'a [f32]) {
    if lag >= 0 {
        let offset = lag as usize;
        let length = clean.len().min(estimate.len().saturating_sub(offset));
        (&clean[..length], &estimate[offset..offset + length])
    } else {
        let offset = (-lag) as usize;
        let length = clean.len().saturating_sub(offset).min(estimate.len());
        (&clean[offset..offset + length], &estimate[..length])
    }
}

/// Find the best SNR over a bounded relative offset using coarse-to-fine
/// normalized-correlation search. This is diagnostic only; zero-lag SNR remains
/// the primary metric. The coarse search keeps runtime practical for long clips.
fn best_lag_snr_db(clean: &[f32], estimate: &[f32], max_lag: usize) -> Option<(f64, isize)> {
    if clean.len() < 2 || estimate.len() < 2 {
        return None;
    }
    let bound = max_lag
        .min(clean.len().saturating_sub(1))
        .min(estimate.len().saturating_sub(1)) as isize;
    let correlation = |lag: isize| -> f64 {
        let (reference, actual) = overlap_for_lag(clean, estimate, lag);
        let mut dot = 0.0f64;
        let mut reference_power = 0.0f64;
        let mut actual_power = 0.0f64;
        for i in (0..reference.len()).step_by(8) {
            let a = reference[i] as f64;
            let b = actual[i] as f64;
            dot += a * b;
            reference_power += a * a;
            actual_power += b * b;
        }
        let denominator = (reference_power * actual_power).sqrt();
        if denominator <= 1e-20 { -1.0 } else { dot / denominator }
    };

    let coarse_step = 48isize.min(bound.max(1));
    let mut best_lag = 0isize;
    let mut best_correlation = f64::NEG_INFINITY;
    let mut lag = -bound;
    while lag <= bound {
        let score = correlation(lag);
        if score > best_correlation {
            best_correlation = score;
            best_lag = lag;
        }
        lag += coarse_step;
    }
    // Always include the positive boundary in case the coarse step does not land on it.
    for candidate in [-bound, bound] {
        let score = correlation(candidate);
        if score > best_correlation {
            best_correlation = score;
            best_lag = candidate;
        }
    }
    let refine_start = (best_lag - coarse_step).max(-bound);
    let refine_end = (best_lag + coarse_step).min(bound);
    for candidate in refine_start..=refine_end {
        let score = correlation(candidate);
        if score > best_correlation {
            best_correlation = score;
            best_lag = candidate;
        }
    }
    let (reference, actual) = overlap_for_lag(clean, estimate, best_lag);
    snr_db(reference, actual).map(|score| (score, best_lag))
}

fn print_snr_diagnostics(label: &str, clean: &[f32], estimate: &[f32], input_snr: f64) -> Result<(), Box<dyn Error>> {
    let zero_lag = snr_db(clean, estimate).ok_or(format!("cannot calculate {label} zero-lag SNR"))?;
    let (best_snr, lag) = best_lag_snr_db(clean, estimate, SAMPLE_RATE as usize / 10)
        .ok_or(format!("cannot calculate {label} lag-adjusted SNR"))?;
    println!(
        "{label}: {zero_lag:.2} dB zero-lag SNR ({:+.2} dB vs noisy input); best bounded diagnostic SNR {best_snr:.2} dB at lag {lag} samples ({:.2} ms)",
        zero_lag - input_snr,
        lag as f64 * 1000.0 / SAMPLE_RATE as f64
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args.len() > 5 {
        return Err("usage: quality_benchmark <clean.wav> <noisy.wav> [model_dir] [output_dir]".into());
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
    print_snr_diagnostics("RNNoise", &clean, &rnnoise, input_snr)?;

    let deepfilter = process_full_stream(
        EngineKind::DeepFilterNet { model_dir: model_dir.clone() },
        &noisy,
    )?;
    print_snr_diagnostics("DeepFilterNet3", &clean, &deepfilter, input_snr)?;
    println!("These scores apply only to this aligned reference pair; listen for speech artifacts as well.");

    if let Some(output_dir) = args.get(4) {
        let output_dir = PathBuf::from(output_dir);
        std::fs::create_dir_all(&output_dir)?;
        let noisy_path = output_dir.join("noisy_input.wav");
        let rn_path = output_dir.join("rnnoise.wav");
        let df_path = output_dir.join("deepfilternet.wav");
        if noisy_path.exists() || rn_path.exists() || df_path.exists() {
            return Err(format!(
                "refusing to overwrite existing benchmark outputs in {}",
                output_dir.display()
            ).into());
        }
        write_mono_wav(&noisy_path, &noisy)?;
        write_mono_wav(&rn_path, &rnnoise)?;
        write_mono_wav(&df_path, &deepfilter)?;
        println!(
            "Wrote listening outputs: {}, {}, and {}",
            noisy_path.display(), rn_path.display(), df_path.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{best_lag_snr_db, snr_db};

    #[test]
    fn snr_metric_rewards_an_exact_clean_signal() {
        let clean = [0.2f32, -0.1, 0.4, 0.3];
        let noisy = [0.3f32, -0.2, 0.3, 0.5];
        assert!(snr_db(&clean, &clean).unwrap().is_infinite());
        assert!(snr_db(&clean, &clean).unwrap() > snr_db(&clean, &noisy).unwrap());
    }

    #[test]
    fn bounded_lag_search_recovers_a_known_offset() {
        let clean = [0.1f32, 0.4, -0.2, 0.7, 0.3, -0.5, 0.2];
        let mut delayed = vec![0.0f32];
        delayed.extend_from_slice(&clean[..clean.len() - 1]);
        let (score, lag) = best_lag_snr_db(&clean, &delayed, 2).unwrap();
        assert_eq!(lag, 1);
        assert!(score.is_infinite());
    }

    #[test]
    fn bounded_lag_search_rejects_empty_inputs() {
        assert!(best_lag_snr_db(&[], &[1.0], 3).is_none());
        assert!(best_lag_snr_db(&[1.0], &[], 3).is_none());
    }

    #[test]
    fn snr_rejects_mismatched_lengths_and_silent_reference() {
        assert!(snr_db(&[1.0], &[1.0, 2.0]).is_none());
        assert!(snr_db(&[0.0, 0.0], &[0.1, 0.2]).is_none());
    }
}
