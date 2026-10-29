# Noise Cancellation Model

**Self-contained. Offline. Models ship in the repo.**

| Engine | Needs files? | Quality | CPU | Status |
|--------|--------------|---------|-----|--------|
| **RNNoise** (default) | **None** – weights embedded | Very good | Extremely low | Production |
| **DeepFilterNet3** | Vendored in `./models` | Excellent | Moderate | True DFN3 DSP |

## Design principle

- Default path (`Denoiser::with_defaults()`) uses **RNNoise** with weights compiled into the binary.
- **No network calls** at runtime.
- DeepFilterNet3 ONNX graphs (`enc.onnx`, `erb_dec.onnx`, `df_dec.onnx`) are **vendored under `models/`**.

## Quick start

```toml
[dependencies]
noise_cancellation = { git = "https://github.com/R-6167/noise-cancellation-model" }
```

```rust
use noise_cancellation::Denoiser;

// Fully offline RNNoise
let mut den = Denoiser::with_defaults()?;
den.process(&input, &mut output)?;
```

### DeepFilterNet3 (local models already in repo)

```toml
noise_cancellation = { git = "https://github.com/R-6167/noise-cancellation-model", features = ["rnnoise", "deepfilternet"] }
```

```rust
let mut den = Denoiser::with_deepfilternet_auto()?; // uses ./models
```

## CLI

```bash
# RNNoise
cargo run --features cli --release --bin noise-cancel -- input.wav out.wav

# DeepFilterNet3
cargo run --features "cli,deepfilternet" --release --bin noise-cancel -- \
  input.wav out.wav --engine deepfilternet --model-dir ./models
```

## CI

- RNNoise: build + tests (offline)
- DeepFilterNet3: build against **vendored** `models/*.onnx` (no download step)
- Lint: fmt + clippy

## License

MIT OR Apache-2.0  
DeepFilterNet3 weights: upstream MIT / Apache-2.0 (Rikorose/DeepFilterNet).
