# DeepFilterNet3 models

## Required files (official export)

| File | Size | Role |
|------|------|------|
| `enc.onnx` | ~1.9 MB | Encoder |
| `erb_dec.onnx` | ~3.2 MB | ERB mask decoder |
| `df_dec.onnx` | ~3.2 MB | Deep-filter coefficient decoder |
| `config.ini` | ~2 KB | DSP hyperparameters |

**RNNoise needs none of these** — its weights are embedded in the binary.

## Vendor into this repo (one-time)

From the repository root:

```bash
bash scripts/vendor_models.sh
git add models/*.onnx models/config.ini
git commit -m "Vendor DeepFilterNet3 ONNX models"
git push
```

Or download the official archive and extract:

```bash
curl -sL https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3_onnx.tar.gz | tar -xz
cp tmp/export/*.onnx models/
cp tmp/export/config.ini models/
```

After the three `.onnx` files are present under `models/`, `Denoiser::with_deepfilternet_auto()` works fully offline.

## License

Upstream DeepFilterNet3: MIT OR Apache-2.0 (Rikorose/DeepFilterNet).
