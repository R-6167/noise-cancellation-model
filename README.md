# Noise Cancellation Model

**One library. Two engines. One clean API.**

| Engine | Quality | CPU | Latency | Status |
|--------|---------|-----|---------|--------|
| **RNNoise** | Very good | Extremely low | ~10 ms | Production ready |
| **DeepFilterNet3** | Excellent | Moderate | ~20–40 ms | **ONNX loading + STFT path live** |

## What’s new in 0.2.1

- Real `ort` environment + `Session::builder().commit_from_file()` loading
- Validates enc.onnx / erb_dec.onnx / df_dec.onnx (or combined graph)
- Graceful fallback to spectral path if ONNX init fails
- Still one unified `Denoiser` for both engines

## Design

```
Denoiser
  +-- RnnoiseEngine
  +-- DeepFilterNetEngine
        +-- STFT / iSTFT (FFT 960, hop 480)
        +-- ONNX sessions (enc + erb_dec + df_dec)
        +-- Spectral fallback (always available)
```

## Quick Start

```toml
[dependencies]
noise_cancellation = { git = "https://github.com/R-6167/noise-cancellation-model", features = ["rnnoise", "deepfilternet"] }
```

```rust
// High-quality path
let mut den = Denoiser::with_deepfilternet("./models")?;
den.process(&input, &mut output)?;
```

Place models in `./models`:
- `enc.onnx` + `erb_dec.onnx` + `df_dec.onnx`
- or `deepfilter.onnx` / `dfn3.onnx`

Download: https://github.com/Rikorose/DeepFilterNet → `models/DeepFilterNet3_onnx.tar.gz`

## CLI

```bash
cargo run --features "cli,deepfilternet" --release --bin noise-cancel -- \
  input.wav out.wav --engine deepfilternet --model-dir ./models --strength 0.9
```

## Roadmap

- [x] Unified Denoiser API
- [x] RNNoise production engine
- [x] DeepFilterNet3 STFT / iSTFT
- [x] ONNX model discovery + session loading via ort
- [ ] Full neural ERB feature extraction + mask/coefficient application
- [ ] C ABI for audio-studio
- [ ] Resampling / multi-channel / plugins

## License

MIT OR Apache-2.0
