# Noise Cancellation Model

**One library. Two engines. One clean API.**

| Engine | Quality | CPU | Latency | Status |
|--------|---------|-----|---------|--------|
| **RNNoise** | Very good | Extremely low | ~10 ms | Production ready |
| **DeepFilterNet3** | Excellent | Moderate (rt-optimised) | ~20–40 ms | **Real-time neural path** |

## What’s new in 0.3.1 – Real-time optimisations

- **Zero heap allocation** on the per-hop hot path after warm-up
- Pre-allocated STFT buffers (`framed`, `time_out`, spectrum, OLA)
- Pre-allocated feature buffers (`erb`, `spec_ch`, `mask`, `coefs`)
- **Cached tensor input/output names** – trial-and-error only on first frames
- ort sessions built with `intra_threads(1)` + `GraphOptimizationLevel::Level3`
- Minimal logging after name cache is resolved

## Pipeline (still the same algorithm)

```
hop
  → STFT (pre-allocated)
  → feat_erb + feat_spec (pre-allocated)
  → encoder.run()          ← cached input names
  → erb_dec.run() → mask   ← cached output name
  → df_dec.run()  → coefs  ← cached output name
  → apply mask + coefs (in-place)
  → residual spectral gate
  → iSTFT + OLA (pre-allocated)
```

## Quick Start

```toml
[dependencies]
noise_cancellation = { git = "https://github.com/R-6167/noise-cancellation-model", features = ["rnnoise", "deepfilternet"] }
```

```rust
let mut den = Denoiser::with_deepfilternet("./models")?;
den.process(&input, &mut output)?;
```

## CLI

```bash
cargo run --features "cli,deepfilternet" --release --bin noise-cancel -- \
  input.wav out.wav --engine deepfilternet --model-dir ./models --strength 0.9
```

## Roadmap

- [x] Unified Denoiser API
- [x] RNNoise production engine
- [x] Full neural path (enc + decoders)
- [x] Real-time optimisations (alloc-free hot path, name cache, single-thread sessions)
- [ ] Exact ERB matrix / norm stats from official config
- [ ] Multi-frame DF FIR with history
- [ ] C ABI for audio-studio
- [ ] Resampling / multi-channel / plugins

## License

MIT OR Apache-2.0
