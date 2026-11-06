"""Native Apple-Silicon environment, MPS without CPU fallback, and no CUDA in the port."""

import platform
import re
from pathlib import Path

import torch

from m3d_lamed_mps.mps_utils import mps_fallback_enabled, require_mps

ROOT = Path(__file__).resolve().parents[1]
PORT_SOURCES = sorted((ROOT / "m3d_lamed_mps").glob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))


def test_native_arm64_python():
    assert platform.system() == "Darwin"
    assert platform.machine() == "arm64", "Python must be a native arm64 build (not Rosetta x86_64)"


def test_mps_available_and_fallback_disabled(mps):
    assert torch.backends.mps.is_built() and torch.backends.mps.is_available()
    assert not mps_fallback_enabled()
    assert require_mps().type == "mps"
    x = torch.ones(4, device=mps) * 2
    assert x.device.type == "mps" and x.sum().item() == 8


def test_mps_working_set_fits_bf16_model(mps):
    # bf16 weights ~7.6 GiB + activations; Metal must allow a working set well above that.
    assert torch.mps.recommended_max_memory() / 2**30 > 12


def test_no_cuda_in_port_sources():
    pattern = re.compile(r"\.cuda\(|torch\.cuda|device\s*=\s*['\"]cuda|['\"]cuda['\"]|autocast\(\s*['\"]cuda")
    hits = []
    for f in PORT_SOURCES:
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{f.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not hits, "CUDA references found:\n" + "\n".join(hits)


def test_runtime_package_has_no_monai_or_einops_imports():
    pattern = re.compile(r"^\s*(import|from)\s+(monai|einops)\b")
    hits = [f.name for f in (ROOT / "m3d_lamed_mps").glob("*.py") if any(pattern.match(l) for l in f.read_text().splitlines())]
    assert not hits, f"runtime modules must not depend on MONAI/einops: {hits}"


# The ONE deliberate exception: the independent reference that runs the official remote code, in the
# separate .venv-ref environment only (it must never import, or be imported by, the port).
REFERENCE_SCRIPT = ROOT / "scripts" / "upstream_reference.py"


def test_never_uses_trust_remote_code():
    hits = [f.name for f in PORT_SOURCES if f != REFERENCE_SCRIPT and re.search(r"trust_remote_code\s*=\s*True|AutoModelForCausalLM", f.read_text())]
    assert not hits, hits


def test_reference_script_is_isolated_from_the_port():
    src = REFERENCE_SCRIPT.read_text()
    assert not re.search(r"^\s*(import|from)\s+(m3d_lamed_mps|_bootstrap)\b", src, re.M), "reference must not use the port"
    users = [f.name for f in PORT_SOURCES if f != REFERENCE_SCRIPT and re.search(r"^\s*(import|from)\s+upstream_reference\b", f.read_text(), re.M)]
    assert not users, f"upstream_reference.py must only run as a subprocess in .venv-ref: {users}"
