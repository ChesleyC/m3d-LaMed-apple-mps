#!/usr/bin/env python
"""Phase-2 numerical validation of the real checkpoint on MPS.

Runs the same MRI + prompt through several (device, dtype) configurations, one after another
(each loads the checkpoint, computes, then frees memory), and compares against a reference:

  * projected image tokens (vision tower + projector output, 256 x 3072)
  * next-token logits after the prompt (prefill)      -> cosine, relative error, top-1 / top-5
  * greedy continuation of N tokens                   -> first divergence position, text

Reference: MPS fp32 by default; --cpu-reference adds a CPU fp32 run (device-independent gold
standard, ~16 GB RAM, slower) and makes it the reference.

    python scripts/validate_numerics.py --image brain.nii.gz --cpu-reference --json-out reports/phase2/numerics.json
"""

import _bootstrap  # noqa: F401

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from m3d_lamed_mps.loading import build_empty_model, load_checkpoint_weights, load_config, load_tokenizer, resolve_model_dir
from m3d_lamed_mps.mps_utils import require_mps, resolve_dtype, synchronize
from m3d_lamed_mps.preprocessing import preprocess_volume
from m3d_lamed_mps.prompts import DEFAULT_CAPTION_QUESTION, build_prompt


def run_config(model_dir, device, dtype, input_ids, image, new_tokens):
    t0 = time.perf_counter()
    cfg = load_config(model_dir)
    model = build_empty_model(cfg, device=device, dtype=dtype)
    load_checkpoint_weights(model, model_dir, verbose=False)
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig.from_pretrained(str(model_dir))
    load_s = time.perf_counter() - t0
    ids, img = input_ids.to(device), torch.as_tensor(image)[None].to(device=device, dtype=dtype)
    with torch.inference_mode():
        t = time.perf_counter()
        feats = model.encode_images(img).float().cpu()
        logits = model(input_ids=ids, images=img, logits_to_keep=1).logits[0, -1].float().cpu()
        gen = model.generate(img, ids, max_new_tokens=new_tokens, do_sample=False, pad_token_id=32000)[0].cpu().tolist()
        synchronize(device)
        compute_s = time.perf_counter() - t
    del model
    gc.collect()
    if torch.device(device).type == "mps":
        torch.mps.empty_cache()
    return {"feats": feats, "logits": logits, "ids": gen, "load_s": round(load_s, 1), "compute_s": round(compute_s, 1)}


def compare(a, ref, tok):
    cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x.flatten().double(), y.flatten().double(), dim=0))  # noqa: E731
    rel = lambda x, y: float((x.double() - y.double()).norm() / y.double().norm())  # noqa: E731
    top5a, top5r = set(a["logits"].topk(5).indices.tolist()), set(ref["logits"].topk(5).indices.tolist())
    n = min(len(a["ids"]), len(ref["ids"]))
    first_div = next((i for i in range(n) if a["ids"][i] != ref["ids"][i]), None)
    return {
        "image_tokens_cos": round(cos(a["feats"], ref["feats"]), 6), "image_tokens_rel_err": round(rel(a["feats"], ref["feats"]), 5),
        "logits_cos": round(cos(a["logits"], ref["logits"]), 6), "logits_rel_err": round(rel(a["logits"], ref["logits"]), 5),
        "top1_equal": int(a["logits"].argmax()) == int(ref["logits"].argmax()), "top5_overlap": len(top5a & top5r),
        "greedy_identical": a["ids"] == ref["ids"], "greedy_first_divergence": first_div,
        "text": tok.decode(a["ids"], skip_special_tokens=True),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--question", default=DEFAULT_CAPTION_QUESTION)
    ap.add_argument("--dtypes", nargs="+", default=["bf16", "fp16"])
    ap.add_argument("--cpu-reference", action="store_true")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    mps = require_mps()
    d = resolve_model_dir(args.model_dir)
    tok = load_tokenizer(d)
    image, _ = preprocess_volume(args.image)  # 1 x 32 x 256 x 256 (channel first); batched in run_config
    ids = tok(build_prompt(args.question), return_tensors="pt")["input_ids"]

    configs = ([("cpu", "fp32")] if args.cpu_reference else []) + [("mps", "fp32")] + [("mps", dt) for dt in args.dtypes]
    results = {}
    for dev, dt in configs:
        name = f"{dev}-{dt}"
        print(f"running {name} ...", flush=True)
        results[name] = run_config(d, mps if dev == "mps" else torch.device("cpu"), resolve_dtype(dt), ids, image, args.new_tokens)
        print(f"  load {results[name]['load_s']}s, compute {results[name]['compute_s']}s: {tok.decode(results[name]['ids'], skip_special_tokens=True)[:120]!r}")

    ref_name = "cpu-fp32" if args.cpu_reference else "mps-fp32"
    report = {"reference": ref_name, "question": args.question, "new_tokens": args.new_tokens, "comparisons": {}}
    print(f"\nreference: {ref_name}")
    for name, r in results.items():
        if name == ref_name:
            continue
        c = compare(r, results[ref_name], tok)
        report["comparisons"][name] = c
        print(f"{name:10s} image-tokens cos {c['image_tokens_cos']:.6f} | logits cos {c['logits_cos']:.6f} rel {c['logits_rel_err']:.4f} | "
              f"top1 {'=' if c['top1_equal'] else '!='} top5 overlap {c['top5_overlap']}/5 | greedy identical {c['greedy_identical']} "
              f"(first divergence {c['greedy_first_divergence']})")
    report["reference_text"] = tok.decode(results[ref_name]["ids"], skip_special_tokens=True)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, indent=1))
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()
