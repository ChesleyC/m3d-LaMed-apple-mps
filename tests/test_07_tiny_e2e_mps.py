"""End-to-end multimodal forward / generate / segmentation on MPS with a tiny random Phi-3.

The vision tower, projector, SegVol and tokenizer have their real sizes; only the decoder is
shrunk (2 layers, hidden 256) so CPU references are fast. Checks the Transformers-5 integration
(generate with inputs_embeds, KV cache, hidden-state semantics, [SEG] extraction) on MPS.
"""

import numpy as np
import pytest
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from m3d_lamed_mps.mps_utils import DeviceAudit
from m3d_lamed_mps.pipeline import M3DLaMedMPS
from m3d_lamed_mps.prompts import DEFAULT_CAPTION_QUESTION, build_prompt
from tests.conftest import cos_sim, rel_err

PAD = 32000


@pytest.fixture(scope="module")
def inputs(tokenizer):
    torch.manual_seed(3)
    ids = tokenizer(build_prompt(DEFAULT_CAPTION_QUESTION), return_tensors="pt")["input_ids"]
    return ids, torch.rand(1, 1, 32, 256, 256)


class ForceTokenAt(LogitsProcessor):
    def __init__(self, token_id, step):
        self.token_id, self.step, self.calls = token_id, step, 0

    def __call__(self, input_ids, scores):
        if self.calls == self.step:
            scores = torch.full_like(scores, -1e4)
            scores[:, self.token_id] = 0
        self.calls += 1
        return scores


def test_forward_logits_mps_vs_cpu(mps, tiny_model_cpu, clone, inputs):
    ids, img = inputs
    with torch.no_grad():
        ref = tiny_model_cpu(input_ids=ids, images=img).logits
        assert ref.shape == (1, ids.shape[1], 32015)
        for dt, max_rel, min_cos in ((torch.float32, 1e-3, 0.99999), (torch.bfloat16, 6e-2, 0.998), (torch.float16, 2e-2, 0.9998)):
            m = clone(tiny_model_cpu, mps, dt)
            out = m(input_ids=ids.to(mps), images=img.to(mps, dt)).logits
            assert out.device.type == "mps" and torch.isfinite(out).all()
            assert rel_err(out, ref) < max_rel and cos_sim(out, ref) > min_cos, (dt, rel_err(out, ref))


def test_image_tokens_are_replaced(tiny_model_cpu, inputs):
    ids, img = inputs
    with torch.no_grad():
        emb = tiny_model_cpu.embed_multimodal_inputs(ids, img)
        text_emb = tiny_model_cpu.model.embed_tokens(ids)
        feats = tiny_model_cpu.encode_images(img)
    assert emb.shape == text_emb.shape
    assert torch.equal(emb[:, 0], text_emb[:, 0]) and torch.equal(emb[:, 257:], text_emb[:, 257:])
    assert torch.equal(emb[:, 1:257], feats)


def test_misplaced_image_tokens_raise(tiny_model_cpu, tokenizer, inputs):
    _, img = inputs
    bad = tokenizer("Describe" + "<im_patch>" * 256, return_tensors="pt")["input_ids"]
    with pytest.raises(ValueError, match="im_patch"):
        tiny_model_cpu.embed_multimodal_inputs(bad, img)


def test_greedy_generate_mps_equals_cpu(mps, tiny_model_cpu, clone, inputs):
    ids, img = inputs
    ref = tiny_model_cpu.generate(img, ids, max_new_tokens=12, do_sample=False, pad_token_id=PAD)
    m = clone(tiny_model_cpu, mps)
    out = m.generate(img.to(mps), ids.to(mps), max_new_tokens=12, do_sample=False, pad_token_id=PAD)
    assert out.device.type == "mps" and out.shape[0] == 1 and 1 <= out.shape[1] <= 12  # new tokens only
    assert out.cpu().tolist() == ref.tolist()


def test_kv_cache_equals_no_cache_on_mps(mps, tiny_model_cpu, clone, inputs):
    ids, img = inputs
    m = clone(tiny_model_cpu, mps)
    a = m.generate(img.to(mps), ids.to(mps), max_new_tokens=10, min_new_tokens=10, do_sample=False, pad_token_id=PAD)
    b = m.generate(img.to(mps), ids.to(mps), max_new_tokens=10, min_new_tokens=10, do_sample=False, pad_token_id=PAD, use_cache=False)
    assert a.tolist() == b.tolist()


def test_eager_vs_sdpa_language_attention(mps, tiny_model_cpu, clone, inputs):
    ids, img = inputs
    m = clone(tiny_model_cpu, mps)
    with torch.no_grad():
        m.set_attn_implementation("sdpa")
        a = m(input_ids=ids.to(mps), images=img.to(mps)).logits
        m.set_attn_implementation("eager")
        b = m(input_ids=ids.to(mps), images=img.to(mps)).logits
        m.set_attn_implementation("sdpa")
    assert rel_err(a, b) < 1e-4


def test_sampling_on_mps(mps, tiny_model_cpu, clone, inputs):
    ids, img = inputs
    m = clone(tiny_model_cpu, mps, torch.bfloat16)
    torch.mps.manual_seed(0)
    out = m.generate(img.to(mps, torch.bfloat16), ids.to(mps), max_new_tokens=8, do_sample=True, top_p=0.9, temperature=1.0, pad_token_id=PAD)
    assert out.shape[1] >= 1


def test_seg_path_matches_manual_computation(mps, tiny_model_cpu, clone, inputs):
    """[SEG] at generation step 3 -> the seg prompt is the last-layer (post-norm) hidden state of the
    token *preceding* [SEG] (upstream: ``output_ids[:, 1:] == seg_token_id``)."""
    ids, img = inputs
    m = clone(tiny_model_cpu, mps)
    proc = LogitsProcessorList([ForceTokenAt(32014, step=3)])
    out_ids, logits = m.generate(img.to(mps), ids.to(mps), seg_enable=True, max_new_tokens=6, min_new_tokens=6,
                                 do_sample=False, pad_token_id=PAD, logits_processor=proc)
    assert out_ids[0, 3].item() == 32014
    assert logits.shape == (1, 1, 32, 256, 256) and torch.isfinite(logits).all()
    with torch.no_grad():
        emb = m.embed_multimodal_inputs(ids.to(mps), img.to(mps))
        full = torch.cat([emb, m.model.embed_tokens(out_ids[:, :-1])], dim=1)
        h = m.model(inputs_embeds=full).last_hidden_state
        h_seg = h[:, ids.shape[1] + 2]  # position of generated token #2, which predicts [SEG] (#3)
        expected = m.model.seg_module(img.to(mps), m.model.seg_projector(h_seg))
    assert rel_err(logits, expected) < 1e-3


def test_seg_without_seg_token_gives_neg_inf(mps, tiny_model_cpu, clone, inputs):
    ids, img = inputs
    m = clone(tiny_model_cpu, mps, torch.bfloat16)
    # forbid [SEG]
    class Ban(LogitsProcessor):
        def __call__(self, input_ids, scores):
            scores[:, 32014] = -1e4
            return scores
    out_ids, logits = m.generate(img.to(mps, torch.bfloat16), ids.to(mps), seg_enable=True, max_new_tokens=4, do_sample=False,
                                 pad_token_id=PAD, logits_processor=LogitsProcessorList([Ban()]))
    assert torch.isneginf(logits).all()


def test_full_generate_runs_only_on_mps(mps, tiny_model_cpu, clone, inputs):
    ids, img = inputs
    m = clone(tiny_model_cpu, mps, torch.bfloat16)
    audit = DeviceAudit()
    with audit:
        m.generate(img.to(mps, torch.bfloat16), ids.to(mps), seg_enable=True, max_new_tokens=6, do_sample=False, pad_token_id=PAD,
                   logits_processor=LogitsProcessorList([ForceTokenAt(32014, step=2)]))
    s = audit.report.summary()
    assert s["ok"], s
    assert s["mps_compute_ops"] > 1000 and not s["cpu_compute_ops"] and not s["other_device_ops"]


def test_pipeline_wrapper(mps, tiny_model_cpu, clone, tokenizer):
    m = clone(tiny_model_cpu, mps, torch.bfloat16)
    pipe = M3DLaMedMPS(m, tokenizer, mps)
    img = np.random.default_rng(0).random((1, 32, 256, 256), dtype=np.float32)
    res = pipe.generate(img, max_new_tokens=8, min_new_tokens=8, audit=True)
    assert res.new_tokens == 8 and isinstance(res.text, str) and res.prompt_tokens == 257 + 15
    t = res.timings
    assert t["decode_tokens"] == 7 and t["time_to_first_token_s"] > 0 and t["vision_encoder_projector_s"] > 0
    assert t["vision_encoder_projector_s"] < t["time_to_first_token_s"] <= t["total_s"]
    assert res.memory["mps_peak_allocated"] > 0 and res.memory["lifetime_max_phys_footprint"] > 0
    assert res.audit["ok"]
    seg = pipe.generate(img, max_new_tokens=4, seg=True)
    assert seg.seg_logits.shape == (1, 1, 32, 256, 256)
