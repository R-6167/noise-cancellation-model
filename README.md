# Noise Cancellation Model

**High-quality, embeddable noise suppression library written in Rust.**

Dual-engine design focused on practical real-world use:

| Engine | Quality | CPU | Latency | Status |
|--------|---------|-----|---------|--------|
| **RNNoise** (default) | Very good | Extremely low | ~10 ms | Ready |
| **DeepFilterNet3** | Excellent (SOTA open) | Moderate | ~20–40 ms | Planned (feature-gated) |

Designed from day one as a **library/plugin** that can be integrated into larger applications (your audio-studio, VoIP clients, DAWs, mobile apps, etc.).

## Features

- Real-time streaming API (10 ms frames)
- Offline batch processing
- Strength / wet-dry control
- Pure-Rust RNNoise via `nnnoiseless` (no C dependencies for the light engine)
- C ABI (`cdylib` + `staticlib`) for easy embedding into C/C++ codebases
- Reasonable CPU usage – RNNoise runs comfortably on low-end devices and phones
- MIT **or** Apache-2.0 licensed

## Quick Start (Library)

```toml
[dependencies]
noise_cancellation = { git = "https://github.com/R-6167/noise-cancellation-model", features = ["rnnoise"] }
```

```rust
use noise_cancellation::{Denoiser, DenoiserConfig, Strength};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut denoiser = Denoiser::with_defaults()?;

    // Mono f32 samples at 48 kHz
    let mut input  = vec![0.0f32; 480 * 100]; // 1 second
    let mut output = vec![0.0f32; input.len()];

    denoiser.process(&input, &mut output)?;
    // ... use cleaned `output`
    Ok(())
}
```

## CLI

```bash
cargo run --features cli --bin noise-cancel -- input.wav output.wav --strength 0.9
```

## Integration into audio-studio (C++ side)

Because the crate builds as both `cdylib` and `staticlib`, you can:

1. Link the static library, or
2. Use the C ABI (coming in the next iteration) from your existing C++ core.

A thin C header will be provided under `bindings/`.

## Roadmap

- [x] RNNoise engine + clean streaming API
- [x] Offline CLI
- [x] Strength / wet-dry mix
- [ ] Full DeepFilterNet3 ONNX engine (feature `deepfilternet`)
- [ ] Official C ABI header + example
- [ ] Optional resampling so any input sample rate works transparently
- [ ] Multi-channel support
- [ ] VST3 / CLAP / LV2 plugin wrappers
- [ ] Fine-tuning / custom model training notes

## Design Principles

- **Voice first** – preserve natural speech character
- **Reasonable CPU** – prefer efficient models; never require a GPU
- **Embeddable** – clean API, no global state, one instance per stream
- **Local-first** – everything runs on-device
- **Evidence-led** – future releases will include PESQ / STOI / SI-SDR numbers on public datasets

## License

MIT OR Apache-2.0
