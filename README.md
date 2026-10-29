# Noise Cancellation Model

**One library. Two engines. One clean API.**

| Engine | Quality | CPU | Latency | Status |
|--------|---------|-----|---------|--------|
| **RNNoise** | Very good | Extremely low | ~10 ms | Production ready |
| **DeepFilterNet3** | Excellent | Moderate | ~20–40 ms | **Full neural path (enc + decoders)** |

## What’s new in 0.3.0

- ERB decoder session kept alive
- DF decoder session kept alive
- Encoder → decoder chaining
- ERB mask extraction + linear-frequency interpolation
- Deep-filter coefficient application
- Residual spectral gate for uncovered bins
- Automatic fallback at every stage

## Pipeline

```
hop
  → STFT (FFT 960, hop 480)
  → feat_erb + feat_spec
  → encoder.run()
  → erb_dec.run()  → 32-band mask  → apply
  → df_dec.run()   → coefficients  → apply
  → residual spectral gate
  → iSTFT + overlap-add
```

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

Models (place in `./models`):
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
- [x] ONNX session loading
- [x] Encoder inference
- [x] ERB + DF decoder inference + mask/coef application
- [ ] Exact ERB matrix / norm stats from official config
- [ ] Multi-frame DF FIR with history
- [ ] C ABI for audio-studio
- [ ] Resampling / multi-channel / plugins

## License

MIT OR Apache-2.0
