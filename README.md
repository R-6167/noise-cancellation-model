# Noise Cancellation Model

**Self-contained. Offline. No model download required for the default engine.**

| Engine | Needs files? | Quality | CPU | Status |
|--------|--------------|---------|-----|--------|
| **RNNoise** (default) | **None** – weights embedded | Very good | Extremely low | Production |
| **DeepFilterNet3** | Optional ONNX in `./models` | Excellent | Moderate | True DFN3 DSP |

## Design principle

The library is **independent and self-contained**:

- Default path (`Denoiser::with_defaults()`) uses **RNNoise** with weights compiled into the binary.
- **No network calls** at runtime. Ever.
- DeepFilterNet3 is opt-in (`feature = "deepfilternet"`) and only reads local ONNX files you already have (or that CI cached).

## Quick start (zero download)

```toml
[dependencies]
noise_cancellation = { git = "https://github.com/R-6167/noise-cancellation-model" }
# default features = ["rnnoise"] — fully offline
```

```rust
use noise_cancellation::Denoiser;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut den = Denoiser::with_defaults()?; // works offline
    let input = vec![0.0f32; 480 * 10];
    let mut output = vec![0.0f32; input.len()];
    den.process(&input, &mut output)?;
    Ok(())
}
```

## Optional: DeepFilterNet3

```toml
noise_cancellation = { git = "https://github.com/R-6167/noise-cancellation-model", features = ["rnnoise", "deepfilternet"] }
```

Place official graphs in `./models` (or set `NOISE_CANCEL_MODEL_DIR`):

- `enc.onnx`, `erb_dec.onnx`, `df_dec.onnx`

```rust
let mut den = Denoiser::with_deepfilternet_auto()?; // local files only
```

CI downloads these once and caches them; the library never fetches them itself.

## CLI

```bash
# RNNoise – no models needed
cargo run --features cli --release --bin noise-cancel -- input.wav out.wav

# DeepFilterNet3 – uses ./models if present
cargo run --features "cli,deepfilternet" --release --bin noise-cancel -- \
  input.wav out.wav --engine deepfilternet --model-dir ./models
```

## CI

GitHub Actions (`.github/workflows/ci.yml`):

- **RNNoise job** – build, example, tests – fully offline
- **DeepFilterNet job** – builds with models cached by Actions
- **Lint** – fmt + clippy

## License

MIT OR Apache-2.0
DeepFilterNet3 weights (when present) follow upstream MIT / Apache-2.0.
