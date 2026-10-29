# Noise Cancellation Model

**Self-contained. Offline. Models ship in the repo. C ABI included.**

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
let mut den = Denoiser::with_defaults()?;
den.process(&input, &mut output)?;
```

## C / C++ integration (plugin ABI)

Header: [`include/noise_cancellation.h`](include/noise_cancellation.h)

```bash
cargo build --release --features "rnnoise,c-api"
# with DeepFilterNet3:
# cargo build --release --features "rnnoise,deepfilternet,c-api"
```

```c
#include "noise_cancellation.h"

NcHandle *h = NULL;
nc_create(NC_ENGINE_RNNOISE, NULL, 0.85f, &h);
nc_process(h, input, output, n_samples);  /* mono f32 @ 48 kHz */
nc_destroy(h);
```

| Function | Purpose |
|----------|--------|
| `nc_create` | Create handle (RNNoise or DeepFilterNet3) |
| `nc_process` | Process mono f32 buffer (in-place OK) |
| `nc_destroy` | Free handle |
| `nc_set_strength` / `nc_get_strength` | 0.0–1.0 mix |
| `nc_reset` | Clear stream state |
| `nc_latency_samples` | Algorithmic latency |
| `nc_engine_name` | `"RNNoise"` / `"DeepFilterNet3"` |

Link `target/release/libnoise_cancellation.a` (or shared `.so` / `.dylib`).  
Example: [`examples/c_host_snippet.c`](examples/c_host_snippet.c).

**Threading:** one `NcHandle` per audio stream; do not call `nc_process` concurrently on the same handle without a lock.

## DeepFilterNet3

```rust
let mut den = Denoiser::with_deepfilternet_auto()?; // uses ./models
```

```c
nc_create(NC_ENGINE_DEEPFILTERNET, "./models", 0.85f, &h);
```

## CLI

```bash
cargo run --features cli --release --bin noise-cancel -- input.wav out.wav
cargo run --features "cli,deepfilternet" --release --bin noise-cancel -- \
  input.wav out.wav --engine deepfilternet --model-dir ./models
```

## CI

- RNNoise: build + tests (offline)
- DeepFilterNet3: build against vendored `models/*.onnx`
- Lint: fmt + clippy

## License

MIT OR Apache-2.0  
DeepFilterNet3 weights: upstream MIT / Apache-2.0 (Rikorose/DeepFilterNet).
