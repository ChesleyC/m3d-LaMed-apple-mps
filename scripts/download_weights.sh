#!/usr/bin/env bash
# Phase 2: download the original M3D-LaMed-Phi-3-4B checkpoint (pinned revision) into models/.
# Exactly the files this port needs (~16.2 GB). The redundant model.bin (16 GB) and mm_projector.bin
# are NOT downloaded. The repository's .py files are fetched for reference only - never imported/executed.
#
#   scripts/download_weights.sh             # -> models/
#   scripts/download_weights.sh --dry-run   # list what would be downloaded, download nothing
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REVISION="329bed510c2cff0ad3fe6f2a1a7075f45e2706e9"
DEST="$PROJECT_DIR/models"
export HF_HOME="$PROJECT_DIR/.hf_home"   # keep Hugging Face caches inside the project

HF="$PROJECT_DIR/.venv/bin/hf"
[ -x "$HF" ] || { echo "hf CLI not found; run scripts/setup_env.sh first"; exit 1; }

"$HF" download GoodBaiBai88/M3D-LaMed-Phi-3-4B \
  model-00001-of-00004.safetensors model-00002-of-00004.safetensors \
  model-00003-of-00004.safetensors model-00004-of-00004.safetensors \
  model.safetensors.index.json config.json generation_config.json \
  tokenizer.model tokenizer_config.json special_tokens_map.json added_tokens.json \
  configuration_m3d_lamed.py modeling_m3d_lamed.py \
  --revision "$REVISION" --local-dir "$DEST" "$@"

if [[ " $* " != *" --dry-run "* ]]; then
  "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/verify_model_dir.py" --model-dir "$DEST" --sha256 --headers
fi
