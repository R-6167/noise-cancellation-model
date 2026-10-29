#!/usr/bin/env bash
# One-time: download official DeepFilterNet3 ONNX graphs into models/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/models"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading DeepFilterNet3_onnx.tar.gz ..."
curl -sL "https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3_onnx.tar.gz" \
  -o "$TMP/dfn3.tar.gz"

tar -xzf "$TMP/dfn3.tar.gz" -C "$TMP"

# Handle common archive layouts
for name in enc.onnx erb_dec.onnx df_dec.onnx config.ini; do
  found="$(find "$TMP" -name "$name" | head -1 || true)"
  if [[ -n "$found" ]]; then
    cp "$found" "$ROOT/models/$name"
    echo "  + models/$name ($(wc -c < "$ROOT/models/$name") bytes)"
  else
    echo "  ! missing $name" >&2
    exit 1
  fi
done

echo "Done. Commit with:"
echo "  git add models/*.onnx models/config.ini && git commit -m 'Vendor DFN3 models' && git push"
