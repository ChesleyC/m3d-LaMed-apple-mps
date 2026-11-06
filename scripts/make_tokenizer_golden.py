"""Regenerate tests/data/golden_tokenization_tf4424.json with the *original* tokenizer stack.

Must run under Transformers 4.42.4 (the version family upstream M3D used), NOT the project venv:

    python3.13 -m venv /tmp/tf442 && /tmp/tf442/bin/pip install "transformers==4.42.4" --no-deps
    /tmp/tf442/bin/pip install tokenizers sentencepiece protobuf regex pyyaml packaging requests tqdm filelock numpy "huggingface_hub==0.23.5" safetensors
    # 4.42 pins numpy<2 (no py3.13 wheels); its import-time version check must be disabled in that throwaway venv:
    sed -i '' 's/^        require_version_core(deps\\[pkg\\])/        pass/' /tmp/tf442/lib/python3.13/site-packages/transformers/dependency_versions_check.py
    HF_HOME=/tmp/hfhome /tmp/tf442/bin/python -I scripts/make_tokenizer_golden.py <project_dir>

Only the slow (pure sentencepiece) LlamaTokenizer is used, so the numpy/tokenizers pins are irrelevant.
"""

import json
import sys
from pathlib import Path

import transformers
from transformers import LlamaTokenizer

project = Path(sys.argv[1]).resolve()
tok_dir = project / "upstream" / "hf_M3D-LaMed-Phi-3-4B"
out_path = project / "tests" / "data" / "golden_tokenization_tf4424.json"

t = LlamaTokenizer.from_pretrained(str(tok_dir), use_fast=False, model_max_length=512, padding_side="right")
src = (project / "m3d_lamed_mps" / "prompts.py").read_text()
ns: dict = {}
exec(src.split("# The official Quickstart question.")[0].replace("from .modeling_m3d_lamed import IMAGE_TOKEN", "IMAGE_TOKEN='<im_patch>'"), ns)
templates = ns["CAPTION_TEMPLATES"]
extra = ["What is liver in this image? Please output the segmentation mask.", "Where is brain in this image? Please output the box.",
         "Is there evidence of hydrocephalus? Choices: A. Yes B. No C. Maybe D. Unknown", "T2/FLAIR hyperintensity, 3.5 x 2.1 cm, in the left frontal lobe.",
         "The [SEG] token and numbers 12345 and émigré unicode.", " leading space question?", "double  space   and\nnewline and\ttab?",
         "trailing space ", "Describe the brain MRI.<|end|>", "中文问题：这是什么？", "▁underscore-like ▁ char", ""]
plain = ["Hello world", " leading space", "multiple   spaces\nnewline", "The ventricles are normal.", "[SEG]", "<|end|>x", "a\n\nb"]
IM = "<im_patch>" * 256
prefix = [1] + [32011] * 256

cases = []
for q in templates + extra:
    for task, p in (("caption", IM + q), ("vqa", IM + " " + q)):
        ids = t(p)["input_ids"]
        assert ids[:257] == prefix
        cases.append({"kind": "prompt", "task": task, "question": q, "ids_after_image_tokens": ids[257:]})
for s in plain + extra:
    cases.append({"kind": "plain", "text": s, "ids": t(s)["input_ids"]})

dec = [[450, 9819, 4495, 29889, 32007, 32000], [29871, 13, 450, 29871, 13, 32014, 29889],
       [1, 32011, 32011, 6028, 366, 32014, 29889, 32000], [8236, 29871, 29871, 2913]]
for s in templates[:10] + extra + plain:
    body = t(s)["input_ids"][1:]
    for tail in ([32000], [32007], [32001], [32014, 29889, 32000], []):
        dec.append(body + tail)
dec.append([1] + [32011] * 4 + t("Can you")["input_ids"][1:] + [32000])
dec.append([13, 13, 450])
decodes = [{"ids": d, "skip_special_tokens": sk, "text": t.decode(d, skip_special_tokens=sk)} for d in dec for sk in (True, False)]

json.dump({"transformers": transformers.__version__, "tokenizer": "LlamaTokenizer(use_fast=False) from revision 329bed5",
           "note": "prompt cases: ids == [1] + [32011]*256 + ids_after_image_tokens",
           "cases": cases, "decodes": decodes}, open(out_path, "w"), ensure_ascii=False, indent=0)
print(f"wrote {out_path}: {len(cases)} encode cases, {len(decodes)} decode cases")
