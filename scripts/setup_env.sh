#!/usr/bin/env bash
# Create the project-local native arm64 environment (.venv) with pinned versions.
#   PYTHON=/path/to/python3.13 scripts/setup_env.sh
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3.13 || command -v python3.12 || command -v python3.11 || true)}"
[ -n "$PYTHON" ] || { echo "need Python >= 3.11 (arm64); set PYTHON=..."; exit 1; }
ARCH=$("$PYTHON" -c "import platform; print(platform.machine())")
[ "$ARCH" = "arm64" ] || { echo "Python at $PYTHON is $ARCH, not arm64 (Rosetta?)"; exit 1; }

"$PYTHON" -m venv "$PROJECT_DIR/.venv"
"$PROJECT_DIR/.venv/bin/python" -m pip install -q --upgrade pip wheel setuptools
"$PROJECT_DIR/.venv/bin/python" -m pip install -q -r "$PROJECT_DIR/requirements-lock.txt"
"$PROJECT_DIR/.venv/bin/python" -m pip install -q --no-deps -e "$PROJECT_DIR"
PYTORCH_ENABLE_MPS_FALLBACK=0 "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/check_env.py"
