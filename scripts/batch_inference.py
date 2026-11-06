#!/usr/bin/env python
"""MRI-to-text over many scans with one model load (MPS only), resumable JSONL output + summary.

    .venv/bin/python scripts/batch_inference.py /path/to/scans/*.nii.gz --jsonl-out /path/to/out/captions.jsonl
    .venv/bin/python scripts/batch_inference.py /path/to/scans --question "Which organ is mainly shown in this image?" --task vqa ...

Each line: file, dataset/contrast (parsed from names like SITE_T1w_..., else empty), question, orientation, text,
token counts, timings and peak memory. Re-running skips (file, question, task, orientation) entries
already present. The summary (<jsonl>.summary.json) has latency/memory statistics, output diversity,
and simple keyword counts (anatomy / modality words) for judging domain shift.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import re
import statistics
import time
import traceback
from collections import Counter
from pathlib import Path

import torch

from m3d_lamed_mps.pipeline import M3DLaMedMPS
from m3d_lamed_mps.preprocessing import ORIENTATIONS, preprocess_volume
from m3d_lamed_mps.prompts import DEFAULT_CAPTION_QUESTION

SUFFIXES = (".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".npy")
KEYWORDS = {
    "brain/head": r"\b(brain|cerebr\w*|cerebell\w*|intracranial|ventric\w*|hemispher\w*|frontal|temporal|parietal|occipit\w*|"
                  r"white matter|grey matter|gray matter|cortex|cortical|pituitar\w*|sella\w*|skull|calvari\w*|head)\b",
    "spine/neck": r"\b(spine|spinal|vertebra\w*|C[1-7]|odontoid|cervical|thoracic spine|lumbar|sacr\w*|neck)\b",
    "chest": r"\b(lung\w*|pulmonar\w*|pleura\w*|chest|thora\w*|mediastin\w*|heart|cardiac|aort\w*)\b",
    "abdomen/pelvis": r"\b(liver|hepat\w*|kidney\w*|renal|pancrea\w*|spleen|splenic|bowel|colon|abdom\w*|pelvi\w*|bladder|"
                      r"gallbladder|celiac|mesenter\w*|adrenal|uter\w*|ovar\w*|prostat\w*)\b",
    "says MRI": r"\b(MRI|MR|T1|T2|FLAIR|DWI|ADC|gadolinium)\b",
    "says CT": r"\b(CT|Hounsfield|attenuation|contrast[- ]enhanced CT)\b",
    "normal/no abnormality": r"\b(normal|unremarkable|no (significant |obvious )?abnormalit\w*)\b",
}


def collect(inputs: list[str]) -> list[Path]:
    files = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.name.endswith(SUFFIXES))
        elif p.exists():
            files.append(p)
    return files


def parse_name(path: Path) -> tuple[str | None, str | None]:
    m = re.match(r"(.+?)_(T1w|T2w|FLAIR|T1|T2|PD)_", path.name)
    return (m.group(1), m.group(2)) if m else (None, None)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    s = {"entries": len(rows), "ok": len(ok), "errors": len(rows) - len(ok)}
    if not ok:
        return s
    for key in ("time_to_first_token_s", "vision_encoder_projector_s", "decode_tokens_per_s", "total_s"):
        xs = [r["timings"][key] for r in ok if r["timings"].get(key) is not None]
        s[key] = {"median": round(statistics.median(xs), 4), "p10": pct(xs, 0.1), "p90": pct(xs, 0.9), "max": max(xs)} if xs else None
    s["new_tokens"] = {"median": statistics.median(r["new_tokens"] for r in ok), "max": max(r["new_tokens"] for r in ok),
                       "hit_max_new_tokens": sum(r["new_tokens"] >= r["max_new_tokens"] for r in ok)}
    for key, name in (("lifetime_max_phys_footprint", "max_process_peak_footprint_GiB"), ("mps_peak_allocated", "max_mps_peak_allocated_GiB"),
                      ("mps_peak_driver_allocated", "max_mps_peak_driver_GiB")):
        xs = [r["memory"][key] for r in ok if r["memory"].get(key) is not None]
        s[name] = round(max(xs) / 2**30, 2) if xs else None
    texts = [r["text"] for r in ok]
    c = Counter(texts)
    s["unique_texts"] = len(c)
    s["most_common_texts"] = [{"count": n, "text": t} for t, n in c.most_common(5)]
    s["keyword_hits"] = {k: sum(bool(re.search(rx, t, flags=re.I if k not in ("says MRI", "says CT") else 0)) for t in texts)
                         for k, rx in KEYWORDS.items()}
    by = {}
    for r in ok:
        key = f"{r.get('dataset')}/{r.get('contrast')}"
        by.setdefault(key, []).append(r["text"])
    s["per_group_unique_texts"] = {k: f"{len(set(v))}/{len(v)}" for k, v in sorted(by.items())}
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="files and/or directories")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--question", default=DEFAULT_CAPTION_QUESTION)
    ap.add_argument("--task", default="caption", choices=["caption", "vqa"])
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--orientation", default="radiological", choices=ORIENTATIONS)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jsonl-out", required=True)
    args = ap.parse_args()

    files = collect(args.inputs)[: args.limit]
    if not files:
        raise SystemExit("no input volumes found")
    out = Path(args.jsonl_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()] if out.exists() else []
    done = {(r["file"], r["question"], r["task"], r["orientation"]) for r in rows}
    todo = [f for f in files if (str(f), args.question, args.task, args.orientation) not in done]
    print(f"{len(files)} volumes, {len(files) - len(todo)} already done, {len(todo)} to run", flush=True)

    if todo:
        pipe = M3DLaMedMPS.from_pretrained(args.model_dir, dtype=args.dtype)
        t_all = time.perf_counter()
        with out.open("a") as fh:
            for i, f in enumerate(todo, 1):
                dataset, contrast = parse_name(f)
                rec = {"file": str(f), "dataset": dataset, "contrast": contrast, "question": args.question, "task": args.task,
                       "orientation": args.orientation, "dtype": args.dtype, "max_new_tokens": args.max_new_tokens}
                try:
                    image, info = preprocess_volume(f, orientation=args.orientation)
                    res = pipe.generate(image, question=args.question, task=args.task, max_new_tokens=args.max_new_tokens)
                    rec.update(text=res.text, new_tokens=res.new_tokens, prompt_tokens=res.prompt_tokens,
                               timings=res.timings, memory=res.memory, crop_box=info.crop_box)
                    print(f"[{i}/{len(todo)}] {f.name}: {res.new_tokens} tok, {res.timings['total_s']:.1f}s | {res.text[:110]!r}", flush=True)
                except Exception as e:  # keep going; record the failure
                    rec.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])
                    print(f"[{i}/{len(todo)}] {f.name}: ERROR {e}", flush=True)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                rows.append(rec)
        print(f"batch time {time.perf_counter() - t_all:.0f}s", flush=True)
        del pipe
        torch.mps.empty_cache()

    sel = [r for r in rows if r["question"] == args.question and r["task"] == args.task and r["orientation"] == args.orientation]
    summary = {"question": args.question, "task": args.task, "orientation": args.orientation, **summarize(sel)}
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
