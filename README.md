# Noise Cancellation Model

**Self-contained. Offline. Models ship in the repo. C ABI included. Multi-rate.**

| Engine | Needs files? | Quality | CPU | Status |
|--------|--------------|---------|-----|--------|
| **RNNoise** (default) | **None** – weights embedded | Very good | Extremely low | **Production** |
| **DeepFilterNet3** | Vendored in `./models` | High (unbenchmarked) | Moderate | **Beta** |

## Status / readiness

| Use case | Ready? |
|----------|--------|
| Rust app, RNNoise, mono / multi-rate | **Yes** |
| C/C++ plugin (RNNoise), multi-channel | **Yes** |
| Offline CLI batch denoise (RNNoise) | **Yes** |
| DeepFilterNet3 for highest quality | **Beta** — compiles and smoke-tests; validate audio quality yourself before shipping |
| crates.io publish / strict API stability | Not yet (0.x) |

CI on this branch covers format, clippy (RNNoise), release builds, and smoke tests for both engines. Prefer RNNoise for production until DFN quality is measured against real noisy speech (e.g. PESQ/STOI or listening tests).

## Quick start (Rust)

```toml
[dependencies]
noise_cancellation = { git = "https://github.com/R-6167/noise-cancellation-model" }
```

```rust
use noise_cancellation::Denoiser;
let mut den = Denoiser::with_defaults()?; // 48 kHz, RNNoise
den.process(&input, &mut output)?;
```

### Features

| Feature | Default | Description |
|---------|---------|-------------|
| `rnnoise` | yes | Lightweight RNNoise engine (embedded weights) |
| `deepfilternet` | no | DeepFilterNet3 via ONNX Runtime + vendored models |
| `c-api` | no | C ABI (`include/noise_cancellation.h`) |
| `cli` | no | `noise-cancel` binary + WAV I/O |

Enable both engines if needed:

```toml
noise_cancellation = { git = "...", features = ["rnnoise", "deepfilternet"] }
```

## Sample rates

Engines always run at **48 kHz**. Other rates are converted automatically with a streaming **linear** resampler (low CPU, not sinc/high-quality):

```rust
let mut den = Denoiser::with_sample_rate(16_000)?;
// or
Denoiser::new(DenoiserConfig { sample_rate: 44_100, ..Default::default() })?;
```

```c
nc_create_with_rate(NC_ENGINE_RNNOISE, NULL, 0.85f, 44100, &h);
```

Common rates: 8 / 16 / 22.05 / 32 / 44.1 / 48 kHz.

## C / C++ integration (plugin ABI)

Header: [`include/noise_cancellation.h`](include/noise_cancellation.h)

```bash
cargo build --release --features "rnnoise,c-api"
```

```c
#include "noise_cancellation.h"

NcHandle *h = NULL;
nc_create(NC_ENGINE_RNNOISE, NULL, 0.85f, &h);                 /* 48 kHz */
nc_create_with_rate(NC_ENGINE_RNNOISE, NULL, 0.85f, 16000, &h); /* 16 kHz */
nc_process(h, input, output, n_samples);
nc_destroy(h);
```

Multi-channel (interleaved):

```c
NcMultiHandle *mh = NULL;
nc_create_multi_with_rate(NC_ENGINE_RNNOISE, NULL, 0.85f, 48000, 2, &mh);
nc_process_interleaved(mh, interleaved_in, interleaved_out, frames);
nc_destroy_multi(mh);
```

Link `target/release/libnoise_cancellation.a` (or `.so` / `.dylib`).

**Threading:** one handle per audio stream. Do not share a handle across threads without external synchronization.

## DeepFilterNet3 (beta)

Models are vendored under [`models/`](models/) (no download required).

```rust
let mut den = Denoiser::with_deepfilternet_auto()?;
```

Or with an explicit directory:

```rust
let mut den = Denoiser::with_deepfilternet("./models")?;
```

Resolution order for `with_deepfilternet_auto`: `NOISE_CANCEL_MODEL_DIR` → `./models` → crate `models/` → `<exe>/models`.

### Known limitations (DFN)

- Smoke-tested only (load, no NaNs, silence, dry-pass at strength 0). **No formal quality benchmark** vs upstream DeepFilterNet or vs RNNoise.
- Depends on `ort` 2.0 RC (API may still move).
- Clippy with `-D warnings` is enforced for RNNoise; DFN is build-tested but not under the same lint gate.
- Prefer listening tests / objective metrics before shipping DFN as the primary engine.

## CLI

```bash
cargo run --features cli --release --bin noise-cancel -- input.wav out.wav
```

## Strength and wet/dry

Default strength is **0.85**. `wet_dry_mix` (default `true`) blends dry input with the enhanced signal according to strength. Set strength to `1.0` for full suppression output only.

```rust
den.set_strength(Strength(0.9));
```

## Latency

- Engine frame size: **480 samples** (10 ms at 48 kHz).
- Host latency is scaled when sample rate ≠ 48 kHz (`latency_samples()` / `nc_latency_samples`).
- Additional delay may come from the linear resampler’s internal state when rates differ.

## Testing

```bash
# RNNoise (default)
cargo test --release --features rnnoise

# DeepFilterNet3 smoke
cargo test --release --features deepfilternet --test dfn3_smoke

# Resample + multi-channel
cargo test --release --features rnnoise --test resample_smoke
cargo test --release --features rnnoise --test multi_channel_smoke
```

## Platform notes

- CI runs on **Ubuntu** only.
- `ort` downloads platform binaries when the `deepfilternet` feature is enabled (`download-binaries`).
- Cross-compile / mobile targets are not covered by CI.

## License

MIT OR Apache-2.0

DeepFilterNet3 ONNX models: upstream [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) (MIT OR Apache-2.0).
