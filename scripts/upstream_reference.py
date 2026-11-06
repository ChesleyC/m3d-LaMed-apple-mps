#!/usr/bin/env python
"""Independent reference: run the OFFICIAL M3D-LaMed remote code on CPU fp32.

Runs ONLY under .venv-ref (scripts/setup_reference_env.sh: torch 2.2.1, transformers 4.42.4, MONAI 1.3.0).
It does not import the port. The model is loaded exactly as in the official Quickstart
(`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` + slow `AutoTokenizer`), except
device_map="cpu" and float32, the precision the official benchmark scripts used.

For every case in the JSON spec written by scripts/compare_upstream.py it saves:
  <name>.json  input_ids (4.42 slow tokenizer), greedy generated ids, decoded text, timings
  <name>.npz   feats        projected image tokens (vision tower + projector), 256 x 3072
               tf_logits    teacher-forced next-token logits over prompt + generated ids (T x vocab):
                            row t predicts generated token t (row 0 = prefill logits)
               seg_logits   SegVol mask logits 32 x 256 x 256 (seg cases only)

    .venv-ref/bin/python scripts/upstream_reference.py --cases reports/phase2/parity/cases.json --out reports/phase2/parity/upstream
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".hf_home"))  # remote-code module cache stays in the project
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import warnings  # noqa: E402

# MONAI 1.3 deprecation notices (pos_embed) print the local site-packages path; they are irrelevant here
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="run only these case names")
    args = ap.parse_args()

    spec = json.loads(Path(args.cases).read_text())
    model_dir = str(PROJECT_DIR / spec["model_dir"])  # spec paths are project-relative (or absolute)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import monai  # noqa: F401  (version record; imported by the remote code anyway)

    versions = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                "monai": monai.__version__, "numpy": np.__version__, "device": "cpu", "dtype": "float32"}
    print("reference stack:", versions, flush=True)

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, model_max_length=512, padding_side="right", use_fast=False, trust_remote_code=True)
    load_s = time.perf_counter() - t0
    versions["attn_implementation"] = getattr(model.config, "_attn_implementation", None)
    versions["model_class"] = f"{type(model).__module__}.{type(model).__name__}"
    versions["tokenizer_class"] = type(tokenizer).__name__
    print(f"loaded official model ({versions['model_class']}, attn={versions['attn_implementation']}) in {load_s:.1f}s", flush=True)

    for case in spec["cases"]:
        if args.only and case["name"] not in args.only:
            continue
        name, seg = case["name"], bool(case.get("seg", False))
        image = torch.from_numpy(np.load(PROJECT_DIR / case["image"]).astype(np.float32)).unsqueeze(0)  # 1 x 1 x 32 x 256 x 256 (Quickstart)
        input_ids = tokenizer(case["prompt"], return_tensors="pt")["input_ids"]
        n_new = int(case.get("max_new_tokens", spec.get("max_new_tokens", 256)))
        rec = {"name": name, "versions": versions, "input_ids": input_ids[0].tolist(), "seg": seg}
        arrays = {}
        with torch.inference_mode():
            t = time.perf_counter()
            arrays["feats"] = model.encode_images(image)[0].float().numpy()
            rec["encode_s"] = round(time.perf_counter() - t, 2)

            t = time.perf_counter()
            if seg:
                gen, seg_logits = model.generate(image, input_ids, seg_enable=True, max_new_tokens=n_new, do_sample=False, pad_token_id=tokenizer.pad_token_id)
                arrays["seg_logits"] = seg_logits[0, 0].float().numpy()
            else:
                gen = model.generate(image, input_ids, seg_enable=False, max_new_tokens=n_new, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            rec["generate_s"] = round(time.perf_counter() - t, 2)
            gen_ids = gen[0].tolist()
            rec["gen_ids"] = gen_ids
            rec["text"] = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # teacher forcing over the reference sequence (no KV cache): logits that predict each generated token
            t = time.perf_counter()
            full = torch.cat([input_ids, torch.tensor([gen_ids], dtype=input_ids.dtype)], dim=1)
            logits = model(images=image, input_ids=full, use_cache=False).logits[0]
            p = input_ids.shape[1]
            arrays["tf_logits"] = logits[p - 1 : p - 1 + len(gen_ids)].float().numpy()
            rec["tf_forward_s"] = round(time.perf_counter() - t, 2)

        np.savez(out / f"{name}.npz", **arrays)
        (out / f"{name}.json").write_text(json.dumps(rec, indent=1))
        print(f"[{name}] {len(gen_ids)} tokens, generate {rec['generate_s']}s: {rec['text'][:160]!r}", flush=True)
        if seg:
            print(f"[{name}] seg mask voxels > 0: {int((arrays['seg_logits'] > 0).sum())}", flush=True)
    print("done ->", out)


if __name__ == "__main__":
    main()
