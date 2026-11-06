"""Strict streaming loader: meta-init -> MPS, every tensor copied, failure on any mismatch."""

import json

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.phi3.modeling_phi3 import Phi3RotaryEmbedding

from m3d_lamed_mps.loading import build_empty_model, check_model_dir, load_checkpoint_weights, load_config
from tests.conftest import UPSTREAM_HF, tiny_config


def _write_sharded(sd, d, n_shards=2):
    keys = sorted(sd)
    weight_map = {}
    for i in range(n_shards):
        name = f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"
        part = {k: sd[k].contiguous() for k in keys[i::n_shards]}
        save_file(part, str(d / name), metadata={"format": "pt"})
        weight_map.update({k: name for k in part})
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


@pytest.fixture()
def tiny_ckpt(tmp_path, tiny_model_cpu):
    sd = {k: v.detach().clone() for k, v in tiny_model_cpu.state_dict().items()}
    _write_sharded(sd, tmp_path)
    return tmp_path, sd


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_streaming_load_to_mps_is_exact(mps, real_config, tiny_ckpt, dtype):
    d, sd = tiny_ckpt
    model = build_empty_model(tiny_config(real_config), device=mps, dtype=dtype)
    rep = load_checkpoint_weights(model, d, strict=True, verbose=False)
    assert rep.ok and rep.loaded == rep.expected == len(sd)
    got = model.state_dict()
    for k, v in sd.items():
        assert got[k].device.type == "mps" and got[k].dtype == dtype, k
        assert torch.equal(got[k].cpu(), v.to(dtype)), k
    # RoPE buffers must stay float32 and equal a fresh computation (not garbage from to_empty, not bf16)
    rope = model.model.rotary_emb
    fresh = Phi3RotaryEmbedding(config=model.config)
    assert rope.inv_freq.dtype == torch.float32 and rope.inv_freq.device.type == "mps"
    assert torch.equal(rope.inv_freq.cpu(), fresh.inv_freq)
    assert all(not p.requires_grad for p in model.parameters()) and not model.training


def test_missing_tensor_fails(mps, real_config, tiny_ckpt):
    d, sd = tiny_ckpt
    victim = "model.vision_tower.vision_tower.cls_token"
    sd = {k: v for k, v in sd.items() if k != victim}
    for f in d.glob("*.safetensors"):
        f.unlink()
    _write_sharded(sd, d)
    model = build_empty_model(tiny_config(real_config), device=mps)
    with pytest.raises(RuntimeError, match="cls_token"):
        load_checkpoint_weights(model, d, verbose=False)


def test_unexpected_and_misshaped_tensors_fail(mps, real_config, tiny_ckpt):
    d, sd = tiny_ckpt
    sd = dict(sd)
    sd["model.extra.weight"] = torch.zeros(3)
    sd["lm_head.weight"] = torch.zeros(5, 5)
    for f in d.glob("*.safetensors"):
        f.unlink()
    _write_sharded(sd, d)
    model = build_empty_model(tiny_config(real_config), device=mps)
    rep = load_checkpoint_weights(model, d, strict=False, verbose=False)
    assert not rep.ok and rep.unexpected == ["model.extra.weight"] and "lm_head.weight" in rep.missing
    assert any("lm_head.weight" in s for s in rep.shape_mismatch)


def test_check_model_dir_reports_missing_weights(tmp_path):
    rep = check_model_dir(tmp_path)
    assert not rep.ok and "model-00001-of-00004.safetensors" in rep.missing and "config.json" in rep.missing


def test_upstream_metadata_dir_is_complete_except_weights():
    rep = check_model_dir(UPSTREAM_HF, require_weights=False, verify_sha256=True)
    assert rep.ok, str(rep)
    assert len(rep.sha256_checked) >= 7


def test_load_config_validates_identity(tmp_path, real_config):
    assert load_config(UPSTREAM_HF).hidden_size == 3072
    tiny_config(real_config).save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="unexpected config"):
        load_config(tmp_path)
