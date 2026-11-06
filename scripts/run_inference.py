#!/usr/bin/env python
"""MRI-to-text with the original M3D-LaMed-Phi-3-4B checkpoint, entirely on Apple MPS.

    python scripts/run_inference.py --image /path/brain_T1w.nii.gz
    python scripts/run_inference.py --image scan.nii.gz --question "Describe the findings of the medical image you see." \
        --dtype bf16 --max-new-tokens 256 --audit --json-out reports/phase2/run.json

Input: NIfTI / DICOM directory / any SimpleITK format, or an already preprocessed 1x32x256x256 .npy.
Decoding defaults follow the official benchmark (greedy, 256 new tokens).
"""

import _bootstrap  # noqa: F401  (must precede torch)

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from m3d_lamed_mps.mps_utils import fmt_gib, process_footprint, require_mps, resolve_dtype, system_info
from m3d_lamed_mps.pipeline import M3DLaMedMPS
from m3d_lamed_mps.preprocessing import ORIENTATIONS, preprocess_volume, save_montage
from m3d_lamed_mps.prompts import DEFAULT_CAPTION_QUESTION


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="NIfTI (.nii/.nii.gz), DICOM dir, .mha/.nrrd, or preprocessed .npy")
    ap.add_argument("--model-dir", default=None, help="checkpoint directory (default: models/)")
    ap.add_argument("--question", default=DEFAULT_CAPTION_QUESTION)
    ap.add_argument("--task", default="caption", choices=["caption", "vqa", "seg"],
                    help="prompt format: caption = no space after image tokens; vqa/seg = one space")
    ap.add_argument("--dtype", default="bf16", help="bf16 (default, as the official Quickstart) | fp16 | fp32")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--do-sample", action="store_true", help="sampling instead of greedy (Quickstart: top_p=0.9, T=1.0)")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--orientation", default="radiological", choices=ORIENTATIONS)
    ap.add_argument("--reverse-slices", action="store_true", help="slice order superior->inferior instead of inferior->superior")
    ap.add_argument("--clip", type=float, nargs=2, default=(0.5, 99.5), metavar=("LO", "HI"), help="intensity percentile clip")
    ap.add_argument("--no-clip", action="store_true", help="plain min-max normalisation (official behaviour for 8-bit images)")
    ap.add_argument("--seg", action="store_true", help="also run SegVol (needs a [SEG] token in the answer, e.g. task=seg)")
    ap.add_argument("--seg-out", default=None, help="save the binary mask (1x32x256x256 .npy) here")
    ap.add_argument("--seg-montage", default=None, help="write a PNG montage of the input with the mask overlaid in red")
    ap.add_argument("--audit", action="store_true", help="record every ATen op and verify it ran on MPS (slower)")
    ap.add_argument("--montage", default=None, help="write a PNG montage of the preprocessed input")
    ap.add_argument("--save-preprocessed", default=None, help="write the 1x32x256x256 input as .npy")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    device, dtype = require_mps(), resolve_dtype(args.dtype)

    t0 = time.perf_counter()
    image, pinfo = preprocess_volume(args.image, orientation=args.orientation, reverse_slices=args.reverse_slices,
                                     clip_percentiles=None if args.no_clip else tuple(args.clip))
    t_pre = time.perf_counter() - t0
    print(f"preprocessed {args.image} -> {image.shape} in {t_pre:.2f}s (crop {pinfo.crop_box}, clip {pinfo.clip_values})")
    if args.montage:
        print("montage:", save_montage(image, args.montage))
    if args.save_preprocessed:
        np.save(args.save_preprocessed, image)

    print(f"loading checkpoint ({args.dtype}) ...")
    pipe = M3DLaMedMPS.from_pretrained(args.model_dir, device=device, dtype=dtype, attn_implementation=args.attn)
    li = pipe.load_info["load"]
    print(f"loaded {li['loaded']}/{li['expected']} tensors ({li['gb_read']} GB, source {li['source_dtypes']} -> {li['target_dtype']}) "
          f"in {pipe.load_info['total_load_seconds']}s; MPS allocated {fmt_gib(torch.mps.current_allocated_memory())}")

    res = pipe.generate(image, question=args.question, task=args.task, max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample, top_p=args.top_p, temperature=args.temperature,
                        seg=args.seg, audit=args.audit, seed=args.seed)

    print("\n=== question ===\n" + args.question)
    print("=== M3D-LaMed answer ===\n" + res.text + "\n")
    t, m = res.timings, res.memory
    print(f"time to first token {t['time_to_first_token_s']}s (vision encoder+projector {t['vision_encoder_projector_s']}s) | "
          f"{res.new_tokens} new tokens | decode {t['decode_tokens_per_s']} tok/s | generate total {t['total_s']}s")
    print(f"memory: MPS peak allocated {fmt_gib(m['mps_peak_allocated'])}, MPS driver peak {fmt_gib(m['mps_peak_driver_allocated'])}, "
          f"process peak footprint {fmt_gib(m.get('lifetime_max_phys_footprint'))}")
    if res.audit is not None:
        a = res.audit
        print(f"device audit: {'PASS' if a['ok'] else 'FAIL'} - {a['mps_compute_ops']} compute ops on MPS, "
              f"CPU compute ops: {a['cpu_compute_ops'] or 'none'}, host<->device copies: {a['host_device_transfers']}")
    if args.seg and res.seg_logits is not None:
        mask = (torch.sigmoid(res.seg_logits.float()) > 0.5).cpu().numpy().astype(np.uint8)
        print(f"segmentation: logits {tuple(res.seg_logits.shape)}, mask voxels {int(mask.sum())}"
              + (" (no [SEG] token generated -> empty)" if torch.isneginf(res.seg_logits).all() else ""))
        if args.seg_out:
            np.save(args.seg_out, mask[0])
        if args.seg_montage:
            print("segmentation montage:", save_montage(image, args.seg_montage, mask=mask[0]))

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        out = {"system": system_info(), "args": vars(args), "preprocess": {**pinfo.as_dict(), "seconds": round(t_pre, 3)},
               "load": pipe.load_info, "result": res.as_dict(), "final_footprint": process_footprint()}
        Path(args.json_out).write_text(json.dumps(out, indent=1, default=str))
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()
