"""The ported model must have exactly the tensors of the released checkpoint (names, shapes).

Uses only the safetensors *headers* (fetched via HTTP range requests, a few KB) - no weights.
"""

import json

import torch

from m3d_lamed_mps.loading import MANIFEST
from m3d_lamed_mps.modeling_m3d_lamed import LamedPhi3ForCausalLM
from tests.conftest import UPSTREAM_HF


def _meta_model(cfg):
    with torch.device("meta"):
        return LamedPhi3ForCausalLM(cfg)


def test_state_dict_matches_checkpoint_exactly(real_config, checkpoint_headers):
    sd = _meta_model(real_config).state_dict()
    assert len(checkpoint_headers) == 617
    assert set(sd) == set(checkpoint_headers), (set(sd) - set(checkpoint_headers), set(checkpoint_headers) - set(sd))
    mism = {k: (list(sd[k].shape), checkpoint_headers[k]["shape"]) for k in sd if list(sd[k].shape) != checkpoint_headers[k]["shape"]}
    assert not mism, mism
    assert {v["dtype"] for v in checkpoint_headers.values()} == {"F32"}


def test_index_json_agrees_with_headers(checkpoint_headers):
    wm = json.loads((UPSTREAM_HF / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(wm) == set(checkpoint_headers)


def test_parameter_count_and_submodules(real_config):
    m = _meta_model(real_config)
    count = lambda mod: sum(p.numel() for p in mod.parameters())
    assert abs(count(m) / 1e9 - 4.049) < 0.001
    assert abs(count(m.model.vision_tower) / 1e6 - 87.39) < 0.01
    assert abs(count(m.model.mm_projector) / 1e6 - 11.80) < 0.01
    assert abs(count(m.model.seg_module) / 1e6 - 117.33) < 0.01
    assert m.lm_head.weight is not m.model.embed_tokens.weight  # tie_word_embeddings = False


def test_only_rope_buffers_are_non_persistent(real_config):
    m = _meta_model(real_config)
    sd = set(m.state_dict())
    assert {n for n, _ in m.named_buffers() if n not in sd} == {"model.rotary_emb.inv_freq", "model.rotary_emb.original_inv_freq"}


def test_config_semantics(real_config):
    c = real_config
    assert c.model_type == "lamed_phi3" and c.vocab_size == 32015 and c.seg_token_id == 32014
    assert list(c.image_size) == [32, 256, 256] and list(c.patch_size) == [4, 16, 16]
    assert c.num_image_tokens == 256
    assert c.rope_parameters["rope_type"] == "default" and c.rope_parameters["rope_theta"] == 10000.0
    assert c.tie_word_embeddings is False


def test_manifest_pins_revision():
    assert MANIFEST["revision"] == (UPSTREAM_HF / "REVISION").read_text().strip()
    total = sum(v["size"] for k, v in MANIFEST["files"].items() if k.endswith(".safetensors"))
    assert total == 16_196_486_352
