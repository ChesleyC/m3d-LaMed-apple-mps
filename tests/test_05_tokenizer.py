"""Tokenisation must equal what upstream used: Transformers-4.42 *slow* LlamaTokenizer
(``use_fast=False``, ``legacy=False``, ``add_prefix_space=True``).

Ground truth: ``tests/data/golden_tokenization_tf4424.json`` was produced by real Transformers 4.42.4
(see scripts/make_tokenizer_golden.py) on the pinned tokenizer files.
"""

import json
from pathlib import Path

import pytest

from m3d_lamed_mps.loading import load_tokenizer
from m3d_lamed_mps.prompts import CAPTION_TEMPLATES, build_prompt
from tests.conftest import UPSTREAM_HF

GOLDEN = json.loads((Path(__file__).parent / "data" / "golden_tokenization_tf4424.json").read_text())
PREFIX = [1] + [32011] * 256


def _cases(kind=None):
    for c in GOLDEN["cases"]:
        if kind and c["kind"] != kind:
            continue
        if c["kind"] == "prompt":
            yield build_prompt(c["question"], task=c["task"]), PREFIX + c["ids_after_image_tokens"]
        else:
            yield c["text"], c["ids"]


@pytest.fixture(scope="module")
def hf_tokenizer():
    return load_tokenizer(UPSTREAM_HF, kind="hf")


def test_golden_is_from_transformers_442():
    assert GOLDEN["transformers"] == "4.42.4" and len(GOLDEN["cases"]) >= 120 and len(GOLDEN["decodes"]) >= 300


def test_encode_matches_transformers_442_exactly(tokenizer):
    bad = [t[-60:] for t, gold in _cases() if tokenizer(t)["input_ids"] != gold]
    assert not bad, bad


def test_decode_matches_transformers_442_exactly(tokenizer):
    bad = []
    for d in GOLDEN["decodes"]:
        if tokenizer.decode(d["ids"], skip_special_tokens=d["skip_special_tokens"]) != d["text"]:
            bad.append(d["ids"][:10])
    assert not bad, bad


def test_hf5_tokenizer_agrees_on_all_official_templates(hf_tokenizer):
    """Transformers 5's LlamaTokenizer is fine for the official prompts (documented divergence elsewhere)."""
    templates = set(CAPTION_TEMPLATES)
    for c in GOLDEN["cases"]:
        if c["kind"] == "prompt" and c["question"] in templates:
            assert hf_tokenizer(build_prompt(c["question"], c["task"]))["input_ids"] == PREFIX + c["ids_after_image_tokens"]


def test_hf5_tokenizer_divergence_is_whitespace_only(hf_tokenizer):
    diffs = [t for t, gold in _cases() if hf_tokenizer(t)["input_ids"] != gold]
    assert diffs, "if this fails, Transformers' LlamaTokenizer now matches 4.42 exactly"
    for t in diffs:
        tail = t.replace("<im_patch>", "")
        assert any(s in tail for s in ("  ", "\n", "\t", "▁", "[SEG]", "<|end|>")) or tail.startswith(" "), repr(tail)


def test_prompt_layout(tokenizer):
    ids = tokenizer(build_prompt(CAPTION_TEMPLATES[0]))["input_ids"]
    assert ids[0] == 1 and ids[1:257] == [32011] * 256 and ids[257] != 32011
    assert tokenizer.convert_ids_to_tokens(ids[257]) == "Can"  # no prefix space after <im_patch>
    vqa = tokenizer(build_prompt("Question?", task="vqa"))["input_ids"]
    assert tokenizer.convert_ids_to_tokens(vqa[257]).startswith("▁")
    t = tokenizer(build_prompt(CAPTION_TEMPLATES[0]), return_tensors="pt")["input_ids"]
    assert tuple(t.shape) == (1, len(ids)) and t.dtype.is_signed


def test_special_ids(tokenizer):
    assert (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) == (1, 32000, 32000)
    assert tokenizer.convert_tokens_to_ids("[SEG]") == 32014 and tokenizer.convert_tokens_to_ids("<im_patch>") == 32011
    assert len(tokenizer) == 32015
    gen = tokenizer("The ventricles are normal.", add_special_tokens=False)["input_ids"] + [32000]
    assert tokenizer.decode(gen, skip_special_tokens=True) == "The ventricles are normal."
