#!/usr/bin/env bash
# Create .venv-ref/: an isolated environment with the ORIGINAL software stack (torch 2.2.1, transformers
# 4.42.4, MONAI 1.3.0, einops) used only by scripts/upstream_reference.py to run the official
# M3D-LaMed remote code on CPU fp32 as an independent reference for the port.
#
#   scripts/setup_reference_env.sh                          # Python 3.11 from conda-forge into .venv-ref/
#   PYTHON=/path/to/python3.11 scripts/setup_reference_env.sh   # or any existing Python 3.10-3.12
#
# ~0.8 GB, git-ignored. The port itself never uses this environment.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV="$PROJECT_DIR/.venv-ref"
cd "$PROJECT_DIR"

if [[ -x "$ENV/bin/python" ]]; then
  echo "reusing $ENV ($("$ENV/bin/python" -V))"
elif [[ -n "${PYTHON:-}" ]]; then
  "$PYTHON" -m venv "$ENV"
else
  CONDA="${CONDA:-$(command -v conda || command -v mamba || true)}"
  [[ -n "$CONDA" ]] || { echo "need conda/mamba (or set PYTHON=/path/to/python3.11)"; exit 1; }
  "$CONDA" create -y -q -p "$ENV" --override-channels -c conda-forge python=3.11 pip
fi

"$ENV/bin/python" - <<'EOF'
import platform, sys
assert platform.machine() == "arm64", f"need native arm64 Python, got {platform.machine()}"
assert (3, 10) <= sys.version_info[:2] <= (3, 12), f"need Python 3.10-3.12 for torch 2.2, got {sys.version}"
EOF
"$ENV/bin/python" -m pip install -q --upgrade pip
"$ENV/bin/python" -m pip install -q -r requirements-ref.txt
"$ENV/bin/python" -c "import torch, transformers, monai, einops, numpy; print('reference env OK: torch', torch.__version__, '| transformers', transformers.__version__, '| monai', monai.__version__, '| numpy', numpy.__version__)"
