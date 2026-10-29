# Noise Cancellation Model

**One library. Two engines. One clean API.**

High-quality, embeddable noise suppression written in Rust.  
RNNoise and DeepFilterNet3 ship together as a single algorithm family selectable at runtime.

| Engine | Quality | CPU cost | Latency | Status |
|--------|---------|----------|---------|--------|
| **RNNoise** | Very good | Extremely low | ~10 ms | Production ready |
| **DeepFilterNet3** | Excellent | Moderate | ~20–40 ms | Streaming path + model loading ready |

Both engines implement the same `NoiseEngine` trait and are used through the identical `Denoiser` interface. Switch engines without changing application code.

## Design

```
Application
    |
    v
Denoiser  (strength, wet/dry, reset, process)
    |
    +-- RnnoiseEngine          (default, ultra-light)
    +-- DeepFilterNetEngine    (high quality, ONNX)
```

- Real-time streaming (10 ms frames)
- Offline batch processing
- Strength + wet/dry mix
- Embeddable (`cdylib` + `staticlib`) for C/C++ hosts (audio-studio, etc.)
- Reasonable CPU – no GPU required
- Local-first, MIT OR Apache-2.0

## Quick Start

```toml
[dependencies]
noise_cancellation = { git = "https://github.com/R-6167/noise-cancellation-model", features = ["rnnoise"] }

# Add high-quality engine:
# features = ["rnnoise", "deepfilternet"]
```

```rust
use noise_cancellation::{Denoiser, DenoiserConfig, Strength};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Default = RNNoise
    let mut den = Denoiser::with_defaults()?;

    let input  = vec![0.0f32; 480 * 100];
    let mut output = vec![0.0f32; input.len()];
    den.process(&input, &mut output)?;
    Ok(())
}
```

### Select DeepFilterNet3

```rust
#[cfg(feature = "deepfilternet")]
{
    let mut den = Denoiser::with_deepfilternet("path/to/models")?;
    // same process() / set_strength() / reset() API
}
```

Place official ONNX files in the model directory:
- `enc.onnx` + `erb_dec.onnx` + `df_dec.onnx`  
  **or**
- `deepfilter.onnx` / `dfn3.onnx`

Sources:
- https://github.com/Rikorose/DeepFilterNet → `models/DeepFilterNet3_onnx.tar.gz`
- Hugging Face mirrors

## CLI

```bash
# RNNoise
cargo run --features cli --release --bin noise-cancel -- input.wav out.wav --strength 0.9

# DeepFilterNet3
cargo run --features "cli,deepfilternet" --release --bin noise-cancel -- \
  input.wav out.wav --engine deepfilternet --model-dir ./models
```

## Integration

The crate builds as `lib` + `cdylib` + `staticlib`.  
One instance of `Denoiser` per continuous stream. Call `reset()` when switching recordings.

## Roadmap

- [x] Unified Denoiser API
- [x] RNNoise production engine
- [x] DeepFilterNet3 model loading + streaming hop processing
- [x] Runtime engine selection (same algorithm surface)
- [ ] Complete DeepFilterNet3 STFT → ERB → deep-filter → iSTFT path
- [ ] C ABI header for audio-studio / C++ hosts
- [ ] Transparent resampling
- [ ] Multi-channel
- [ ] Plugin wrappers (VST3 / CLAP / LV2)

## License

MIT OR Apache-2.0
