# Noise Cancellation Model

**One library. Two engines. One clean API.**

| Engine | Quality | CPU | Latency | Status |
|--------|---------|-----|---------|--------|
| **RNNoise** | Very good | Extremely low | ~10 ms | Production ready |
| **DeepFilterNet3** | Excellent | Moderate | ~30–40 ms | **True DFN3 DSP path** |

## What’s new in 0.4.0 – True DeepFilterNet3

- Official ERB band widths (32 bands, sum = 481 bins)
- Forward + inverse ERB filterbank
- Exponential mean normalization on ERB dB features (α = 0.99)
- Unit normalization on complex spectral features (first 96 bins)
- Order-5 deep-filter FIR with spectrum history buffer
- DF applied on noisy history; high bands from ERB mask
- Attenuation limit support
- Combined (`deepfilter.onnx`) or split (`enc` / `erb_dec` / `df_dec`) graphs
- Strength blends neural output with dry signal

## Pipeline

```
hop (480 @ 48 kHz)
  → STFT 960 + Vorbis window
  → power → ERB analyze → dB → EMA normalize     → feat_erb
  → complex[:96] → unit normalize                → feat_spec
  → encoder (+ erb_dec + df_dec  or  combined)
  → ERB mask → linear-freq gains
  → DF FIR (order 5) on spectrum history
  → fuse (DF low bins + ERB high bins) × strength
  → iSTFT + OLA
```

## Models

Place in `./models`:

| Layout | Files |
|--------|--------|
| Combined | `deepfilter.onnx` or `dfn3.onnx` |
| Split | `enc.onnx` + `erb_dec.onnx` + `df_dec.onnx` |

Sources:
- https://github.com/Rikorose/DeepFilterNet → `models/DeepFilterNet3_onnx.tar.gz`
- https://huggingface.co/soniqo/DeepFilterNet3-ONNX

## Quick Start

```toml
[dependencies]
noise_cancellation = { git = "https://github.com/R-6167/noise-cancellation-model", features = ["rnnoise", "deepfilternet"] }
```

```rust
let mut den = Denoiser::with_deepfilternet("./models")?;
den.set_strength(0.9.into());
den.process(&input, &mut output)?;
```

## CLI

```bash
cargo run --features "cli,deepfilternet" --release --bin noise-cancel -- \
  input.wav out.wav --engine deepfilternet --model-dir ./models --strength 0.9
```

## Remaining for full production

- [ ] C ABI header for audio-studio
- [ ] Transparent resampling (non-48 kHz)
- [ ] Automated parity tests vs official libDF
- [ ] Multi-channel

## License

MIT OR Apache-2.0
