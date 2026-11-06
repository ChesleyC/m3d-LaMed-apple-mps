import os

# Must be set before torch initialises MPS: unsupported MPS ops must raise, never fall back to CPU.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import copy  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_HF = ROOT / "upstream" / "hf_M3D-LaMed-Phi-3-4B"
HEADERS_DIR = UPSTREAM_HF / "safetensors_headers"
UPSTREAM_REPO = ROOT / "upstream" / "M3D"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test")


@pytest.fixture(scope="session")
def mps():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    from m3d_lamed_mps.mps_utils import require_mps

    return require_mps()


@pytest.fixture(scope="session")
def real_config():
    from m3d_lamed_mps.configuration_m3d_lamed import LamedPhi3Config

    if not (UPSTREAM_HF / "config.json").exists():
        pytest.skip("run scripts/fetch_upstream.sh first")
    return LamedPhi3Config.from_pretrained(str(UPSTREAM_HF))


@pytest.fixture(scope="session")
def tokenizer():
    from m3d_lamed_mps.loading import load_tokenizer

    return load_tokenizer(UPSTREAM_HF)


@pytest.fixture(scope="session")
def checkpoint_headers():
    if not HEADERS_DIR.exists():
        pytest.skip("safetensors headers missing; run scripts/fetch_upstream.sh")
    hdr = {}
    for f in sorted(HEADERS_DIR.glob("*.json")):
        h = json.loads(f.read_text())
        h.pop("__metadata__", None)
        hdr.update(h)
    return hdr


def tiny_config(real_config, **overrides):
    """Real vision/SegVol/tokenizer dims, tiny Phi-3 decoder (fast, random weights)."""
    from m3d_lamed_mps.configuration_m3d_lamed import LamedPhi3Config

    d = real_config.to_dict()
    d.update(num_hidden_layers=2, hidden_size=256, intermediate_size=512, num_attention_heads=4, num_key_value_heads=4)
    d.update(overrides)
    return LamedPhi3Config(**d)


@pytest.fixture(scope="session")
def tiny_model_cpu(real_config):
    from m3d_lamed_mps.loading import init_random_weights_
    from m3d_lamed_mps.modeling_m3d_lamed import LamedPhi3ForCausalLM

    torch.manual_seed(0)
    m = LamedPhi3ForCausalLM(tiny_config(real_config)).eval()
    init_random_weights_(m, std=0.05, seed=0)
    # non-zero position embeddings / cls token so the vision path is exercised meaningfully
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith(("position_embeddings", "cls_token")):
                p.normal_(0, 0.02)
    return m


@pytest.fixture()
def clone():
    return lambda m, device, dtype=torch.float32: copy.deepcopy(m).to(device=device, dtype=dtype).eval()


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().float().cpu().flatten(), b.detach().float().cpu().flatten()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))
