#!/usr/bin/env bash
# Re-create upstream/ (reference material, no weights):
#   upstream/hf_M3D-LaMed-Phi-3-4B/  config + tokenizer + original remote code at the pinned revision,
#                                    plus the JSON headers of the 4 safetensors shards (HTTP range requests, ~80 KB)
#   upstream/M3D/                    official BAAI-DCAI/M3D repository at the pinned commit
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REV="329bed510c2cff0ad3fe6f2a1a7075f45e2706e9"
M3D_COMMIT="63e72005a2304eda74579870239bfc7b04b0edae"
HFD="$PROJECT_DIR/upstream/hf_M3D-LaMed-Phi-3-4B"
BASE="https://huggingface.co/GoodBaiBai88/M3D-LaMed-Phi-3-4B/resolve/$REV"
PY="$PROJECT_DIR/.venv/bin/python"

mkdir -p "$HFD/safetensors_headers"
for f in README.md added_tokens.json config.json configuration_m3d_lamed.py generation_config.json \
         model.safetensors.index.json modeling_m3d_lamed.py special_tokens_map.json tokenizer.model tokenizer_config.json; do
  curl -fsSL -o "$HFD/$f" "$BASE/$f"
done
echo "$REV" > "$HFD/REVISION"

for i in 1 2 3 4; do
  f="model-0000$i-of-00004.safetensors"
  n=$(curl -fsSL -r 0-7 "$BASE/$f" | "$PY" -c "import struct,sys;print(struct.unpack('<Q',sys.stdin.buffer.read(8))[0])")
  curl -fsSL -r "8-$((8 + n - 1))" -o "$HFD/safetensors_headers/$f.header.json" "$BASE/$f"
done

if [ ! -d "$PROJECT_DIR/upstream/M3D/.git" ]; then
  git clone -q https://github.com/BAAI-DCAI/M3D.git "$PROJECT_DIR/upstream/M3D"
fi
git -C "$PROJECT_DIR/upstream/M3D" fetch -q --depth 1 origin "$M3D_COMMIT" && git -C "$PROJECT_DIR/upstream/M3D" checkout -q "$M3D_COMMIT"

echo "upstream reference material ready in $PROJECT_DIR/upstream"
