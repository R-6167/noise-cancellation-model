# DeepFilterNet3 models (vendored)

These official ONNX graphs ship with the repository. **No download is required.**

| File | Size | Role |
|------|------|------|
| `enc.onnx` | ~1.9 MB | Encoder |
| `erb_dec.onnx` | ~3.2 MB | ERB mask decoder |
| `df_dec.onnx` | ~3.2 MB | Deep-filter coefficient decoder |
| `config.ini` | ~2 KB | DSP hyperparameters |

## Usage

```rust
// Auto-discovers ./models (or NOISE_CANCEL_MODEL_DIR / crate models / <exe>/models)
let mut den = Denoiser::with_deepfilternet_auto()?;
```

**RNNoise** (default engine) does not use these files — its weights are embedded in the binary.

## Source

Upstream: [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) — `DeepFilterNet3_onnx.tar.gz`  
License: MIT OR Apache-2.0
