"""Rehearsal of the Phase-2 command-line scripts on a tiny synthetic checkpoint (no real weights).

Runs every script as a subprocess exactly as a user would, so that Phase 2 cannot fail on plumbing.
``M3D_LAMED_TEST_MODE=1`` lets the loader accept the tiny (non-4B) config and file sizes.
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import SimpleITK as sitk

from tests.conftest import ROOT, UPSTREAM_HF, tiny_config
from tests.test_06_loader import _write_sharded

PY = sys.executable


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory, real_config, tiny_model_cpu):
    d = tmp_path_factory.mktemp("tiny_ckpt")
    tiny_config(real_config).save_pretrained(d)
    for f in ("generation_config.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"):
        shutil.copy(UPSTREAM_HF / f, d / f)
    _write_sharded({k: v.detach().clone() for k, v in tiny_model_cpu.state_dict().items()}, d)
    return d


@pytest.fixture(scope="module")
def mri(tmp_path_factory):
    d = tmp_path_factory.mktemp("mri")
    z, y, x = np.mgrid[0:40, 0:64, 0:56].astype(np.float32)
    vol = np.where(((z - 20) / 18) ** 2 + ((y - 32) / 28) ** 2 + ((x - 28) / 25) ** 2 <= 1, 300.0, 0.0).astype(np.float32)
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((1.0, 1.0, 1.5))
    sitk.WriteImage(sitk.DICOMOrient(img, "RAS"), str(d / "brain.nii.gz"))
    return d / "brain.nii.gz"


def _run(args, check=True):
    env = {**os.environ, "M3D_LAMED_TEST_MODE": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "0"}
    p = subprocess.run([PY, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    if check and p.returncode != 0:
        raise AssertionError(f"{args[0]} failed ({p.returncode}):\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    return p


def test_check_env():
    assert "OK: MPS ready" in _run(["scripts/check_env.py"]).stdout


def test_preprocess_cli(tmp_path, mri):
    _run(["scripts/preprocess_mri.py", str(mri), "-o", str(tmp_path / "x.npy"), "--montage", str(tmp_path / "x.png")])
    v = np.load(tmp_path / "x.npy")
    assert v.shape == (1, 32, 256, 256) and (tmp_path / "x.png").exists()


def test_run_inference_cli(mps, tmp_path, tiny_dir, mri):
    out = tmp_path / "run.json"
    p = _run(["scripts/run_inference.py", "--model-dir", str(tiny_dir), "--image", str(mri), "--max-new-tokens", "8",
              "--audit", "--seg", "--seg-out", str(tmp_path / "mask.npy"), "--seg-montage", str(tmp_path / "seg.png"),
              "--montage", str(tmp_path / "m.png"), "--json-out", str(out)])
    assert "=== M3D-LaMed answer ===" in p.stdout and "device audit: PASS" in p.stdout
    r = json.loads(out.read_text())
    assert r["load"]["load"]["ok"] and r["result"]["audit"]["ok"] and r["result"]["new_tokens"] <= 8
    assert r["result"]["seg_logits_shape"] == [1, 1, 32, 256, 256] and np.load(tmp_path / "mask.npy").shape == (1, 32, 256, 256)
    seg_png = sitk.ReadImage(str(tmp_path / "seg.png"))
    assert seg_png.GetNumberOfComponentsPerPixel() == 3 and seg_png.GetSize() == (8 * 256, 2 * 256)


def test_benchmark_cli(mps, tmp_path, tiny_dir, mri):
    out = tmp_path / "b.json"
    _run(["scripts/benchmark.py", "--model-dir", str(tiny_dir), "--image", str(mri), "--runs", "1", "--warmup", "0",
          "--max-new-tokens", "4", "--json-out", str(out)])
    s = json.loads(out.read_text())["summary"]
    assert s["weights"] == "pretrained" and s["median_total_s"] > 0 and s["peak_mps_allocated_GiB"] >= 0


def test_validate_numerics_cli(mps, tmp_path, tiny_dir, mri):
    out = tmp_path / "n.json"
    _run(["scripts/validate_numerics.py", "--model-dir", str(tiny_dir), "--image", str(mri), "--cpu-reference",
          "--dtypes", "bf16", "--new-tokens", "4", "--json-out", str(out)])
    r = json.loads(out.read_text())
    assert r["reference"] == "cpu-fp32" and set(r["comparisons"]) == {"mps-fp32", "mps-bf16"}
    c = r["comparisons"]["mps-fp32"]
    assert c["image_tokens_cos"] > 0.9999 and c["logits_cos"] > 0.9999 and c["top1_equal"]


def test_verify_model_dir_rejects_incomplete(tiny_dir):
    p = _run(["scripts/verify_model_dir.py", "--model-dir", str(tiny_dir)], check=False)
    assert p.returncode == 1 and "NOT READY" in p.stdout


def test_batch_inference_cli_resumes(mps, tmp_path, tiny_dir, mri):
    d = tmp_path / "scans"
    d.mkdir()
    for name in ("SITEA_T1w_0001_x.nii.gz", "SITEB_T2w_0002_y.nii.gz"):
        shutil.copy(mri, d / name)
    out = tmp_path / "batch.jsonl"
    args = ["scripts/batch_inference.py", str(d), "--model-dir", str(tiny_dir), "--max-new-tokens", "4", "--jsonl-out", str(out)]
    p = _run(args)
    assert "2 to run" in p.stdout
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2 and all("error" not in r for r in rows)
    assert {(r["dataset"], r["contrast"]) for r in rows} == {("SITEA", "T1w"), ("SITEB", "T2w")}
    s = json.loads((tmp_path / "batch.jsonl.summary.json").read_text())
    assert s["ok"] == 2 and s["errors"] == 0 and s["max_process_peak_footprint_GiB"] > 0 and s["decode_tokens_per_s"]["median"] > 0
    assert "0 to run" in _run(args).stdout and len(out.read_text().splitlines()) == 2  # resume: nothing re-run


def test_shell_scripts_are_executable():
    for sh in sorted((ROOT / "scripts").glob("*.sh")):
        assert os.access(sh, os.X_OK), f"{sh.name} is not executable (chmod +x)"
        assert sh.read_text().startswith("#!/usr/bin/env bash"), sh.name
