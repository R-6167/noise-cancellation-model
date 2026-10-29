# Noise Cancellation Model

**One library. Two engines. One clean API.**

High-quality, embeddable noise suppression written in Rust.  
RNNoise and DeepFilterNet3 ship together as a single algorithm family selectable at runtime.

| Engine | Quality | CPU cost | Latency | Status |
|--------|---------|----------|---------|--------|
| **RNNoise** | Very good | Extremely low | ~10 ms | Production ready |
| **DeepFilterNet3** | Excellent | Moderate | ~20–40 ms | **STFT + spectral path live** (full neural next) |

Both engines implement the same `NoiseEngine` trait and are used through the identical `Denoiser` interface.

## What’s new in 0.2.0

- Real STFT / iSTFT (FFT 960, hop 480, Vorbis-style window)
- Overlap-add streaming
- Frequency-dependent spectral suppression on the DeepFilterNet3 path
- Strength control affects the spectral engine
- Unified algorithm surface unchanged – still one `Denoiser`

## Design

```
Application
    |
    v
Denoiser  (strength, wet/dry, reset, process)
    |
    +-- RnnoiseEngine           (default, ultra-light)
    +-- DeepFilterNetEngine     (STFT → spectral suppress → iSTFT)
```

## Quick Start

```toml
[dependencies]
noise_cancellation = { git = "https://github.com/R-6167/noise-cancellation-model", features = ["rnnoise"] }

# High-quality engine:
# features = ["rnnoise", "deepfilternet"]
```

```rust
use noise_cancellation::{Denoiser, Strength};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut den = Denoiser::with_defaults()?;
    let input  = vec![0.0f32; 480 * 100];
    let mut output = vec![0.0f32; input.len()];
    den.process(&input, &mut output)?;
    Ok(())
}
```

### DeepFilterNet3 path

```rust
#[cfg(feature = "deepfilternet")]
{
    let mut den = Denoiser::with_deepfilternet("./models")?;
    den.set_strength(Strength(0.9));
    den.process(&input, &mut output)?;
}
```

Model files (for future full neural inference):
- `enc.onnx` + `erb_dec.onnx` + `df_dec.onnx` **or** `deepfilter.onnx` / `dfn3.onnx`
- Download: https://github.com/Rikorose/DeepFilterNet → `models/DeepFilterNet3_onnx.tar.gz`

Even without the neural weights fully connected, the STFT spectral path already performs real noise reduction.

## CLI

```bash
# RNNoise
cargo run --features cli --release --bin noise-cancel -- input.wav out.wav --strength 0.9

# DeepFilterNet3 spectral path
cargo run --features "cli,deepfilternet" --release --bin noise-cancel -- \
  input.wav out.wav --engine deepfilternet --model-dir ./models --strength 0.9
```

## Roadmap

- [x] Unified Denoiser API
- [x] RNNoise production engine
- [x] DeepFilterNet3 STFT / iSTFT / overlap-add streaming
- [x] Spectral suppression path (audible enhancement)
- [ ] Full neural ERB + deep-filter coefficient application via ONNX
- [ ] C ABI for audio-studio / C++ hosts
- [ ] Transparent resampling
- [ ] Multi-channel / plugin wrappers

## License

MIT OR Apache-2.0
