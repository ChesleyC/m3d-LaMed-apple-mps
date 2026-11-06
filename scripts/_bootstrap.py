"""Common script setup: strict MPS (no CPU fallback), project on sys.path, quiet logs.

Import this module *before* torch in every script.
"""

import os
import sys
import warnings
from pathlib import Path

# Unsupported MPS operators must raise instead of silently running on the CPU.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
# Keep any Hugging Face cache inside the project (self-contained).
PROJECT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".hf_home"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")
warnings.filterwarnings("ignore", category=FutureWarning)

UPSTREAM_HF = PROJECT_DIR / "upstream" / "hf_M3D-LaMed-Phi-3-4B"
REPORTS_DIR = PROJECT_DIR / "reports"
