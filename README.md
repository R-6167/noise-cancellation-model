# Noise Cancellation Model

**One library. Two engines. One clean API.**

| Engine | Quality | CPU | Latency | Status |
|--------|---------|-----|---------|--------|
| **RNNoise** | Very good | Extremely low | ~10 ms | Production ready |
| **DeepFilterNet3** | Excellent | Moderate (rt-optimised) | ~20–40 ms | Real-time neural path |

## What’s new in 0.3.2 – Audio buffer pool

- `AudioBufferPool` + `PooledBuffer` utility
- Both engines acquire work buffers from the pool instead of allocating every `process()` call
- Buffers automatically returned on drop
- Pre-warmed pool sized for streaming 10 ms hops
- Public API: `noise_cancellation::{AudioBufferPool, PooledBuffer}`

## Design

```
Denoiser
  +-- RnnoiseEngine      -- uses AudioBufferPool for work buffer
  +-- DeepFilterNetEngine -- uses AudioBufferPool + pre-alloc STFT/features
```

## Quick Start

```toml
[dependencies]
noise_cancellation = { git = "https://github.com/R-6167/noise-cancellation-model", features = ["rnnoise", "deepfilternet"] }
```

```rust
use noise_cancellation::{Denoiser, AudioBufferPool};

let mut den = Denoiser::with_defaults()?;
den.process(&input, &mut output)?;

// Or use the pool directly in your own audio graph:
let mut pool = AudioBufferPool::for_streaming();
let mut buf = pool.acquire(480 * 10);
buf.extend_from_slice(&samples);
```

## CLI

```bash
cargo run --features "cli,deepfilternet" --release --bin noise-cancel -- \
  input.wav out.wav --engine deepfilternet --model-dir ./models --strength 0.9
```

## Roadmap

- [x] Unified Denoiser API
- [x] RNNoise + DeepFilterNet3 neural path
- [x] Real-time optimisations (alloc-free hot path, name cache)
- [x] Audio buffer pool for work/remainder buffers
- [ ] Exact ERB matrix / norm stats from official config
- [ ] Multi-frame DF FIR with history
- [ ] C ABI for audio-studio
- [ ] Resampling / multi-channel / plugins

## License

MIT OR Apache-2.0
