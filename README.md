# Noise Cancellation Model

**Self-contained. Offline. Models ship in the repo. C ABI included. Multi-rate.**

| Engine | Needs files? | Quality | CPU | Status |
|--------|--------------|---------|-----|--------|
| **RNNoise** (default) | **None** – weights embedded | Very good | Extremely low | Production |
| **DeepFilterNet3** | Vendored in `./models` | Excellent | Moderate | True DFN3 DSP |

## Quick start (Rust)

```toml
[dependencies]
noise_cancellation = { git = "https://github.com/R-6167/noise-cancellation-model" }
```

```rust
use noise_cancellation::Denoiser;
let mut den = Denoiser::with_defaults()?; // 48 kHz
den.process(&input, &mut output)?;
```

## Sample rates

Engines always run at **48 kHz**. Other rates are converted automatically:

```rust
let mut den = Denoiser::with_sample_rate(16_000)?;
// or
Denoiser::new(DenoiserConfig { sample_rate: 44_100, ..Default::default() })?;
```

```c
nc_create_with_rate(NC_ENGINE_RNNOISE, NULL, 0.85f, 44100, &h);
```

Common rates: 8 / 16 / 22.05 / 32 / 44.1 / 48 kHz. Streaming linear resampler (low CPU).

## C / C++ integration (plugin ABI)

Header: [`include/noise_cancellation.h`](include/noise_cancellation.h)

```bash
cargo build --release --features "rnnoise,c-api"
```

```c
#include "noise_cancellation.h"

NcHandle *h = NULL;
nc_create(NC_ENGINE_RNNOISE, NULL, 0.85f, &h);              /* 48 kHz */
nc_create_with_rate(NC_ENGINE_RNNOISE, NULL, 0.85f, 16000, &h); /* 16 kHz */
nc_process(h, input, output, n_samples);
nc_destroy(h);
```

Link `target/release/libnoise_cancellation.a` (or `.so` / `.dylib`).

## DeepFilterNet3

```rust
let mut den = Denoiser::with_deepfilternet_auto()?;
```

## CLI

```bash
cargo run --features cli --release --bin noise-cancel -- input.wav out.wav
```

## License

MIT OR Apache-2.0
