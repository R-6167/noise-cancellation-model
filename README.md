# Noise Cancellation Model

**One library. Two engines. One clean API.**

| Engine | Quality | CPU | Latency | Status |
|--------|---------|-----|---------|--------|
| **RNNoise** | Very good | Extremely low | ~10 ms | Production ready |
| **DeepFilterNet3** | Excellent | Moderate | ~20–40 ms | **Encoder inference live** |

## What’s new in 0.2.2

- Encoder session kept alive across hops
- ERB feature extraction (`feat_erb` [1,1,1,32])
- Spectral feature extraction (`feat_spec` [1,2,1,96])
- Real `session.run()` attempts with common input-name variants
- Automatic fallback to spectral path when tensor names don’t match
- Input/output name logging for easy debugging against your export

## Design

```
Denoiser
  +-- RnnoiseEngine
  +-- DeepFilterNetEngine
        +-- STFT / iSTFT
        +-- ERB + spectral feature builder
        +-- ONNX encoder (live inference)
        +-- Spectral fallback
```

## Quick Start

```toml
[dependencies]
noise_cancellation = { git = "https://github.com/R-6167/noise-cancellation-model", features = ["rnnoise", "deepfilternet"] }
```

```rust
let mut den = Denoiser::with_deepfilternet("./models")?;
den.process(&input, &mut output)?;
```

Models:
- `enc.onnx` + `erb_dec.onnx` + `df_dec.onnx`  or  `deepfilter.onnx` / `dfn3.onnx`
- Download: https://github.com/Rikorose/DeepFilterNet → `models/DeepFilterNet3_onnx.tar.gz`

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
- [x] Encoder feature extraction + inference
- [ ] ERB decoder + DF decoder + mask/coefficient application
- [ ] C ABI for audio-studio
- [ ] Resampling / multi-channel / plugins

## License

MIT OR Apache-2.0
