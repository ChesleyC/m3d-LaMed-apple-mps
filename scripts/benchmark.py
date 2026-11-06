#!/usr/bin/env python
"""Latency / memory benchmark of M3D-LaMed-Phi-3-4B on MPS.

Phase 1 (no checkpoint):   --synthetic   full-size model with random weights (same compute & memory)
Phase 2 (real checkpoint): --model-dir models

Components:
  full    vision encoder + projector + Phi-3 prefill + decode (+ SegVol with --seg)   [default]
  vision  3D ViT + projector only (synthetic weights)
  segvol  SegVol segmentation module only (synthetic weights)

Examples:
  python scripts/benchmark.py --synthetic --dtype bf16 --max-new-tokens 256
  python scripts/benchmark.py --model-dir models --image sample.nii.gz --runs 3
"""

import _bootstrap  # noqa: F401  (must precede torch)

import argparse
import gc
import json
import statistics
import time

import numpy as np
import torch

from m3d_lamed_mps.loading import build_empty_model, init_random_weights_, load_config, load_pretrained, load_tokenizer
from m3d_lamed_mps.mps_utils import (GiB, PeakMemorySampler, fmt_gib, mps_memory, process_footprint, require_mps,
                                     resolve_dtype, synchronize, system_info)
from m3d_lamed_mps.pipeline import M3DLaMedMPS
from m3d_lamed_mps.preprocessing import preprocess_volume


def param_bytes(m: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in m.parameters())


def bench_component(name, dtype, device, runs, warmup):
    from m3d_lamed_mps.modeling_m3d_lamed import ViT3DTower, build_mm_projector
    from m3d_lamed_mps.segvol import build_segmentation_module

    cfg = load_config(_bootstrap.UPSTREAM_HF)
    if name == "vision":
        mods = torch.nn.ModuleList([ViT3DTower(cfg), build_mm_projector(cfg)])
        fn = lambda x: mods[1](mods[0](x))  # noqa: E731
    else:
        mods = torch.nn.ModuleList([build_segmentation_module(cfg)])
        txt = torch.randn(1, 768, device=device, dtype=dtype)
        fn = lambda x: mods[0](x, text_emb=txt)  # noqa: E731
    mods = mods.to(device=device, dtype=dtype).eval()
    init_random_weights_(mods)
    x = torch.rand(1, 1, 32, 256, 256, device=device, dtype=dtype)
    times = []
    with torch.inference_mode(), PeakMemorySampler() as s:
        for i in range(warmup + runs):
            synchronize(device)
            t = time.perf_counter()
            y = fn(x)
            synchronize(device)
            if i >= warmup:
                times.append(time.perf_counter() - t)
    return {"component": name, "dtype": str(dtype), "output_shape": list(y.shape), "params_M": round(sum(p.numel() for p in mods.parameters()) / 1e6, 2),
            "latency_s": {"median": round(statistics.median(times), 4), "min": round(min(times), 4), "runs": [round(t, 4) for t in times]},
            "memory": {**s.result(), **process_footprint()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--synthetic", action="store_true", help="random weights, no checkpoint needed")
    src.add_argument("--model-dir", default=None, help="directory with the checkpoint (default: models/)")
    ap.add_argument("--component", choices=["full", "vision", "segvol"], default="full")
    ap.add_argument("--dtype", default="bf16", help="bf16 | fp16 | fp32")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--image", default=None, help="NIfTI/DICOM/npy; default: random 1x32x256x256")
    ap.add_argument("--question", default=None)
    ap.add_argument("--task", default="caption", choices=["caption", "vqa", "seg"], help="prompt format (seg/vqa: space after the image tokens)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--min-new-tokens", type=int, default=None, help="force decode length (default: = max for --synthetic)")
    ap.add_argument("--seg", action="store_true", help="also run the SegVol branch")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--audit", action="store_true", help="op-level device audit on the first timed run (slower)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    device, dtype = require_mps(), resolve_dtype(args.dtype)
    report = {"system": system_info(), "args": vars(args)}

    if args.component != "full":
        report["result"] = bench_component(args.component, dtype, device, args.runs, args.warmup)
        print(json.dumps(report["result"], indent=1))
    else:
        t0 = time.perf_counter()
        if args.synthetic:
            cfg = load_config(_bootstrap.UPSTREAM_HF)
            model = build_empty_model(cfg, device=device, dtype=dtype, attn_implementation=args.attn)
            init_random_weights_(model)
            from transformers import GenerationConfig
            model.generation_config = GenerationConfig.from_pretrained(str(_bootstrap.UPSTREAM_HF))
            pipe = M3DLaMedMPS(model, load_tokenizer(_bootstrap.UPSTREAM_HF), device)
            load_info = {"weights": "synthetic (random)", "seconds": round(time.perf_counter() - t0, 2)}
            min_new = args.min_new_tokens or args.max_new_tokens
        else:
            model, tok, load_info = load_pretrained(args.model_dir, device=device, dtype=dtype, attn_implementation=args.attn)
            pipe = M3DLaMedMPS(model, tok, device)
            min_new = args.min_new_tokens
        synchronize(device)
        gc.collect()
        report["load"] = {**load_info, "weights_bytes": param_bytes(model), "after_load": {**mps_memory(), **process_footprint()}}
        print(f"model ready in {time.perf_counter() - t0:.1f}s | weights {fmt_gib(param_bytes(model))} | "
              f"MPS allocated {fmt_gib(torch.mps.current_allocated_memory())}")

        if args.image:
            image, pinfo = preprocess_volume(args.image)
            report["preprocess"] = pinfo.as_dict()
        else:
            image = np.random.default_rng(0).random((1, 32, 256, 256), dtype=np.float32)
        kw = dict(max_new_tokens=args.max_new_tokens, min_new_tokens=min_new, seg=args.seg, task=args.task)
        if args.question:
            kw["question"] = args.question

        runs = []
        for i in range(args.warmup + args.runs):
            res = pipe.generate(image, audit=(args.audit and i == args.warmup), **kw)
            tag = "warmup" if i < args.warmup else f"run{i - args.warmup + 1}"
            t = res.timings
            print(f"[{tag}] ttft {t['time_to_first_token_s']:.3f}s (vision {t['vision_encoder_projector_s']:.3f}s) | "
                  f"decode {t['decode_tokens']} tok @ {t['decode_tokens_per_s']} tok/s | total {t['total_s']:.2f}s | "
                  f"MPS peak {fmt_gib(res.memory['mps_peak_allocated'])} | footprint peak {fmt_gib(res.memory.get('lifetime_max_phys_footprint'))}")
            if i >= args.warmup:
                runs.append(res.as_dict())
        if args.audit and runs:
            print("device audit:", json.dumps(runs[0]["audit"], indent=1))
        med = lambda k: round(statistics.median(r["timings"][k] for r in runs), 4)  # noqa: E731
        report["runs"] = runs
        report["summary"] = {
            "dtype": str(dtype), "weights": "synthetic" if args.synthetic else "pretrained",
            "prompt_tokens": runs[0]["prompt_tokens"], "new_tokens": runs[0]["new_tokens"],
            "median_time_to_first_token_s": med("time_to_first_token_s"),
            "median_vision_encoder_projector_s": med("vision_encoder_projector_s"),
            "median_decode_tokens_per_s": med("decode_tokens_per_s"),
            "median_total_s": med("total_s"),
            **({"median_segmentation_s": med("segmentation_s")} if args.seg and all("segmentation_s" in r["timings"] for r in runs) else {}),
            "weights_GiB": round(param_bytes(model) / GiB, 2),
            "peak_mps_allocated_GiB": round(max(r["memory"]["mps_peak_allocated"] for r in runs) / GiB, 2),
            "peak_mps_driver_GiB": round(max(r["memory"]["mps_peak_driver_allocated"] for r in runs) / GiB, 2),
            "process_lifetime_peak_footprint_GiB": round(process_footprint()["lifetime_max_phys_footprint"] / GiB, 2),
        }
        print(json.dumps(report["summary"], indent=1))

    if args.json_out:
        _bootstrap.REPORTS_DIR.mkdir(exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=1, default=str)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()
