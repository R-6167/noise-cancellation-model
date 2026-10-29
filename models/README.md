# DeepFilterNet3 models (optional)

**You do not need these files for the default engine.**
RNNoise ships with weights embedded in the binary and works offline with zero downloads.

If you enable the `deepfilternet` feature, place here:

- `enc.onnx`
- `erb_dec.onnx`
- `df_dec.onnx`

Or set `NOISE_CANCEL_MODEL_DIR`. The library never phones home at runtime.

CI downloads and caches these models automatically.
