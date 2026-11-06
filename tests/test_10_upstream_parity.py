"""Port vs the OFFICIAL implementation, on a tiny random checkpoint (no real weights needed).

The verbatim upstream remote code (`modeling_m3d_lamed.py`, loaded with trust_remote_code) runs in the
reference environment `.venv-ref` (torch 2.2.1 / transformers 4.42.4 / MONAI 1.3.0, CPU fp32) via
scripts/upstream_reference.py; scripts/compare_upstream.py then runs the port on CPU and MPS and
compares tokenizer ids, image tokens, teacher-forced logits, greedy output and the SegVol mask.
This proves the port re-implements the official architecture/weight semantics, not just the
tensor names. Skipped when `.venv-ref` has not been created (scripts/setup_reference_env.sh).
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import SimpleITK as sitk

from tests.conftest import ROOT, UPSTREAM_HF
from tests.test_06_loader import _write_sharded

REF_PY = ROOT / ".venv-ref" / "bin" / "python"
pytestmark = pytest.mark.skipif(not REF_PY.exists(), reason="reference env .venv-ref missing (scripts/setup_reference_env.sh)")


SEG_ID, BIAS_DIM = 32014, 7


def _seg_biased(sd: dict) -> dict:
    """Make the tiny model emit [SEG] every step so the SegVol branch produces a real mask on both sides:
    +5 offset on one embedding dimension -> final norm passes it -> lm_head maps it only to [SEG]."""
    sd = {k: v.clone() for k, v in sd.items()}
    sd["model.embed_tokens.weight"][:, BIAS_DIM] += 5.0
    sd["model.norm.weight"][BIAS_DIM] = 1.0
    sd["lm_head.weight"][:, BIAS_DIM] = 0.0
    sd["lm_head.weight"][SEG_ID, BIAS_DIM] = 20.0
    return sd


@pytest.fixture(scope="module", params=["plain", "seg_biased"])
def tiny_upstream_dir(request, tmp_path_factory, tiny_model_cpu):
    d = tmp_path_factory.mktemp(f"tiny_upstream_{request.param}")
    cfg = json.loads((UPSTREAM_HF / "config.json").read_text())  # genuine 4.42-format config with auto_map
    tc = tiny_model_cpu.config
    cfg.update({k: getattr(tc, k) for k in ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads")})
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    for f in ("generation_config.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
              "configuration_m3d_lamed.py", "modeling_m3d_lamed.py"):
        shutil.copy(UPSTREAM_HF / f, d / f)
    sd = {k: v.detach().clone() for k, v in tiny_model_cpu.state_dict().items()}
    _write_sharded(_seg_biased(sd) if request.param == "seg_biased" else sd, d)
    return request.param, d


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    d = tmp_path_factory.mktemp("parity_inputs")
    z, y, x = np.mgrid[0:40, 0:64, 0:56].astype(np.float32)
    vol = np.where(((z - 20) / 18) ** 2 + ((y - 32) / 28) ** 2 + ((x - 28) / 25) ** 2 <= 1, 300.0 + 3 * x, 0.0).astype(np.float32)
    sitk.WriteImage(sitk.DICOMOrient(sitk.GetImageFromArray(vol), "RAS"), str(d / "mri.nii.gz"))
    rng = np.random.default_rng(0)
    np.save(d / "ct.npy", rng.random((1, 32, 256, 256), dtype=np.float32))
    return d / "mri.nii.gz", d / "ct.npy"


def test_port_matches_official_code(mps, tmp_path, tiny_upstream_dir, inputs):
    variant, ckpt = tiny_upstream_dir
    mri, ct = inputs
    out = tmp_path / "parity"
    env = {**os.environ, "M3D_LAMED_TEST_MODE": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "0", "HF_HOME": str(tmp_path / "hf_home")}
    p = subprocess.run([sys.executable, "scripts/compare_upstream.py", "--model-dir", str(ckpt), "--image", str(mri),
                        "--ct-example", str(ct), "--configs", "cpu-fp32", "mps-fp32", "--max-new-tokens", "12", "--out", str(out)],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, f"{p.stdout[-4000:]}\n{p.stderr[-4000:]}"
    rep = json.loads((out / "parity.json").read_text())
    assert rep["reference"]["transformers"].startswith("4.42") and rep["reference"]["monai"] == "1.3.0"
    for cfg, cases in rep["results"].items():
        assert set(cases) == {"image_caption", "image_vqa_organ", "ct_caption", "ct_seg_liver"}
        for name, r in cases.items():
            assert r["input_ids_equal"], (cfg, name)
            assert r["decode_equal_on_ref_ids"], (cfg, name)
            assert r["image_tokens_cos"] > 0.99999 and r["image_tokens_rel_err"] < 1e-4, (cfg, name, r)
            assert r["tf_logits_cos_min"] > 0.99999 and r["tf_logits_rel_err"] < 1e-4, (cfg, name, r)
            assert r["tf_argmax_agreement"] == 1.0 and r["greedy_identical"], (cfg, name, r)
        seg = cases["ct_seg_liver"]
        if variant == "seg_biased":  # a real mask from the official SegVol path, reproduced by the port
            assert seg["gen_tokens"] == 12 and seg["seg_voxels_ref"] > 1000, seg
            assert seg["seg_dice_vs_ref"] > 0.999 and seg["seg_logits_rel_err"] < 1e-3, seg
