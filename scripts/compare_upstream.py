#!/usr/bin/env python
"""Port vs OFFICIAL implementation on the same inputs (the strongest correctness check).

1. builds the test cases: the given volume (caption + VQA; an MRI/CT file or an official example .npy)
   and an official M3D example CT (caption + liver segmentation), preprocessed once to .npy so both
   sides see bit-identical inputs;
2. runs scripts/upstream_reference.py in .venv-ref (official remote code, torch 2.2.1 /
   transformers 4.42.4 / MONAI 1.3.0, CPU fp32)  [skip with --skip-reference if already done];
3. runs the port for each --configs entry (default mps-fp32 mps-bf16) and compares:
     * input_ids                    port tokenizer vs the original 4.42 slow tokenizer
     * image tokens                 vision tower + projector output (cosine, rel. error)
     * teacher-forced logits        both models scored on the SAME token sequence (the reference's
                                    greedy output): argmax agreement, cosine, rel. error per position
     * free-running greedy output   identical token sequence? first divergence; text
     * SegVol mask (seg case)       Dice of the binary masks, logit rel. error

    .venv/bin/python scripts/compare_upstream.py --image upstream/M3D/Data/data/examples/example_05.npy
    .venv/bin/python scripts/compare_upstream.py --image /path/to/your_scan.nii.gz --out /path/to/output
"""

import _bootstrap  # noqa: F401

import argparse
import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from m3d_lamed_mps.loading import display_path, load_pretrained, resolve_model_dir
from m3d_lamed_mps.mps_utils import require_mps, synchronize
from m3d_lamed_mps.preprocessing import preprocess_volume
from m3d_lamed_mps.prompts import DEFAULT_CAPTION_QUESTION, build_prompt

PROJECT_DIR = _bootstrap.PROJECT_DIR
DEFAULT_CT = PROJECT_DIR / "upstream" / "M3D" / "Data" / "data" / "examples" / "example_03.npy"
VQA_QUESTION = "Which organ is mainly shown in this image?"
SEG_QUESTION = "What is liver in this image? Please output the segmentation mask."  # official README example


def build_cases(args, out: Path) -> dict:
    inp = out / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    img_npy = inp / "image.npy"
    image, info = preprocess_volume(args.image)
    np.save(img_npy, image.astype(np.float32))
    cases = [
        {"name": "image_caption", "image": display_path(img_npy), "prompt": build_prompt(DEFAULT_CAPTION_QUESTION, "caption"), "seg": False},
        {"name": "image_vqa_organ", "image": display_path(img_npy), "prompt": build_prompt(VQA_QUESTION, "vqa"), "seg": False},
    ]
    ct = Path(args.ct_example)
    if ct.exists():
        ct_npy = inp / "ct_example.npy"
        shutil.copyfile(ct, ct_npy)
        cases += [
            {"name": "ct_caption", "image": display_path(ct_npy), "prompt": build_prompt(DEFAULT_CAPTION_QUESTION, "caption"), "seg": False},
            {"name": "ct_seg_liver", "image": display_path(ct_npy), "prompt": build_prompt(SEG_QUESTION, "seg"), "seg": True},
        ]
    else:
        print(f"note: official example CT not found ({ct}); run scripts/fetch_upstream.sh for the CT cases")
    info = info.as_dict()
    info["source"] = display_path(info["source"])
    return {"model_dir": display_path(resolve_model_dir(args.model_dir)), "max_new_tokens": args.max_new_tokens,
            "image_source": display_path(args.image), "image_preprocess": info, "ct_source": display_path(ct), "cases": cases}


def run_reference(args, cases_path: Path, ref_dir: Path) -> None:
    py = Path(args.ref_python)
    if not py.exists():
        sys.exit(f"reference interpreter {py} not found: run scripts/setup_reference_env.sh first")
    cmd = [display_path(py), "scripts/upstream_reference.py", "--cases", display_path(cases_path), "--out", display_path(ref_dir)]
    print("running official reference (CPU fp32):", " ".join(cmd), flush=True)
    t = time.perf_counter()
    subprocess.run([str(py), *cmd[1:]], check=True, cwd=PROJECT_DIR)  # relative paths in cases.json resolve from here
    print(f"reference done in {time.perf_counter() - t:.0f}s", flush=True)


def _cos_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=-1)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm())


def compare_case(model, tok, device, dtype, case, ref, ref_arr, max_new_tokens: int) -> dict:
    ids = tok(case["prompt"], return_tensors="pt")["input_ids"]
    img = torch.from_numpy(np.load(PROJECT_DIR / case["image"]).astype(np.float32))[None].to(device=device, dtype=dtype)
    ids_d = ids.to(device)
    res = {"input_ids_equal": ids[0].tolist() == ref["input_ids"], "prompt_tokens": ids.shape[1]}
    with torch.inference_mode():
        feats = model.encode_images(img)[0].float().cpu()
        rf = torch.from_numpy(ref_arr["feats"])
        res["image_tokens_cos"] = round(float(_cos_rows(feats.flatten()[None], rf.flatten()[None])), 6)
        res["image_tokens_rel_err"] = round(_rel(feats, rf), 6)

        # teacher forcing on the reference's token sequence
        ref_gen = ref["gen_ids"]
        full = torch.cat([ids, torch.tensor([ref_gen], dtype=ids.dtype)], dim=1).to(device)
        p = ids.shape[1]
        logits = model(input_ids=full, images=img, use_cache=False).logits[0, p - 1 : p - 1 + len(ref_gen)].float().cpu()
        rl = torch.from_numpy(ref_arr["tf_logits"])
        rows_cos = _cos_rows(logits, rl)
        port_arg, ref_arg = logits.argmax(-1), rl.argmax(-1)
        top5p, top5r = set(logits[0].topk(5).indices.tolist()), set(rl[0].topk(5).indices.tolist())
        res.update({
            "tf_positions": len(ref_gen),
            "tf_argmax_agreement": round(float((port_arg == ref_arg).float().mean()), 4),
            "tf_argmax_disagree_positions": (port_arg != ref_arg).nonzero().flatten().tolist()[:20],
            "tf_logits_cos_mean": round(float(rows_cos.mean()), 6),
            "tf_logits_cos_min": round(float(rows_cos.min()), 6),
            "tf_logits_rel_err": round(_rel(logits, rl), 6),
            "tf_logits_max_abs_diff": round(float((logits - rl).abs().max()), 4),
            "prefill_top1_equal": int(port_arg[0]) == int(ref_arg[0]),
            "prefill_top5_overlap": len(top5p & top5r),
        })

        # free-running greedy generation
        t = time.perf_counter()
        if case["seg"]:
            gen, seg_logits = model.generate(img, ids_d, seg_enable=True, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=32000)
        else:
            gen = model.generate(img, ids_d, seg_enable=False, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=32000)
        synchronize(device)
        res["generate_s"] = round(time.perf_counter() - t, 2)
        gen_ids = gen[0].cpu().tolist()
        n = min(len(gen_ids), len(ref_gen))
        first_div = next((i for i in range(n) if gen_ids[i] != ref_gen[i]), None if len(gen_ids) == len(ref_gen) else n)
        res.update({"greedy_identical": gen_ids == ref_gen, "greedy_first_divergence": first_div,
                    "gen_tokens": len(gen_ids), "ref_tokens": len(ref_gen),
                    "text": tok.decode(gen_ids, skip_special_tokens=True),
                    "decode_equal_on_ref_ids": tok.decode(ref_gen, skip_special_tokens=True) == ref["text"]})
        if case["seg"]:
            ps = seg_logits[0, 0].float().cpu()
            rs = torch.from_numpy(ref_arr["seg_logits"])
            pm, rm = ps > 0, rs > 0
            inter, tot = int((pm & rm).sum()), int(pm.sum() + rm.sum())
            finite = torch.isfinite(ps) & torch.isfinite(rs)
            res.update({"seg_dice_vs_ref": round(2 * inter / tot, 5) if tot else 1.0,
                        "seg_voxels_port": int(pm.sum()), "seg_voxels_ref": int(rm.sum()),
                        "seg_logits_rel_err": round(_rel(ps[finite], rs[finite]), 6) if finite.any() else None})
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="volume for the caption/VQA cases (NIfTI/DICOM, or a preprocessed 1x32x256x256 .npy)")
    ap.add_argument("--ct-example", default=str(DEFAULT_CT), help="official M3D example CT (.npy, 1x32x256x256)")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--configs", nargs="+", default=["mps-fp32", "mps-bf16"], help="device-dtype, e.g. mps-fp32 mps-bf16 mps-fp16 cpu-fp32")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", default=str(PROJECT_DIR / "reports" / "phase2" / "parity"))
    ap.add_argument("--ref-python", default=str(PROJECT_DIR / ".venv-ref" / "bin" / "python"))
    ap.add_argument("--skip-reference", action="store_true", help="reuse existing reference outputs in <out>/upstream")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    ref_dir = out / "upstream"
    cases_path = out / "cases.json"
    spec = build_cases(args, out)
    cases_path.write_text(json.dumps(spec, indent=1))
    if not args.skip_reference:
        run_reference(args, cases_path, ref_dir)

    refs = {}
    for c in spec["cases"]:
        rj, rn = ref_dir / f"{c['name']}.json", ref_dir / f"{c['name']}.npz"
        if not (rj.exists() and rn.exists()):
            sys.exit(f"missing reference output for {c['name']} in {ref_dir}")
        refs[c["name"]] = (json.loads(rj.read_text()), dict(np.load(rn)))
    ref_versions = next(iter(refs.values()))[0]["versions"]

    mps = require_mps()
    report = {"reference": {"implementation": "official remote code (trust_remote_code)", **ref_versions},
              "image_source": spec["image_source"], "ct_source": spec["ct_source"], "max_new_tokens": args.max_new_tokens,
              "reference_texts": {k: v[0]["text"] for k, v in refs.items()}, "results": {}}
    for cfg in args.configs:
        dev_name, dt = cfg.split("-")
        device = mps if dev_name == "mps" else torch.device("cpu")
        print(f"\n=== port {cfg} ===", flush=True)
        model, tok, info = load_pretrained(args.model_dir, device=device, dtype=dt, verbose=False)
        dtype = next(model.parameters()).dtype
        report["results"][cfg] = {}
        for c in spec["cases"]:
            ref, ref_arr = refs[c["name"]]
            r = compare_case(model, tok, device, dtype, c, ref, ref_arr, spec["max_new_tokens"])
            report["results"][cfg][c["name"]] = r
            line = (f"[{c['name']:14s}] ids {'=' if r['input_ids_equal'] else '!='} | img cos {r['image_tokens_cos']:.6f} | "
                    f"TF argmax {r['tf_argmax_agreement']*100:6.2f}% cos {r['tf_logits_cos_mean']:.6f} (min {r['tf_logits_cos_min']:.5f}) rel {r['tf_logits_rel_err']:.2e} | "
                    f"greedy {'IDENTICAL' if r['greedy_identical'] else 'diverges@' + str(r['greedy_first_divergence'])} ({r['gen_tokens']}/{r['ref_tokens']} tok)")
            if c["seg"]:
                line += f" | seg Dice {r['seg_dice_vs_ref']:.4f} ({r['seg_voxels_port']}/{r['seg_voxels_ref']} vox)"
            print(line, flush=True)
            if not r["greedy_identical"]:
                print(f"    port: {r['text'][:200]!r}\n    ref : {ref['text'][:200]!r}")
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    print("\nreference (official, CPU fp32) outputs:")
    for k, v in report["reference_texts"].items():
        print(f"  {k}: {v!r}")
    jo = Path(args.json_out) if args.json_out else out / "parity.json"
    jo.parent.mkdir(parents=True, exist_ok=True)
    jo.write_text(json.dumps(report, indent=1))
    print("wrote", jo)


if __name__ == "__main__":
    main()
