#!/usr/bin/env bash
# Phase 2 (pretrained weights in models/ + one 3D brain MRI): the complete verification protocol.
#
#   scripts/run_phase2.sh /path/to/brain_T1w.nii.gz ["optional question"]
#
# 1. verify checkpoint files (sizes + sha256 + every tensor name/shape in the shard headers)
# 2. preprocess the MRI and write montages for visual orientation QA (radiological + RAS)
# 3. MRI-to-text on MPS (bf16, greedy, <=256 tokens) with the op-level device audit
# 4. latency / peak memory: bf16 (1 warmup + 3 runs), fp16, fp32, and bf16 with 256 forced tokens
# 5. numerical validation vs CPU-fp32 and MPS-fp32 references (bf16, fp16)
# 6. parity with the OFFICIAL code (torch 2.2.1 / transformers 4.42.4 / MONAI 1.3.0, CPU fp32) on the
#    MRI and an official example CT incl. liver segmentation  [needs scripts/setup_reference_env.sh]
# 7. segmentation on the official example CT (bf16, audit, mask overlay montage) + its latency
# 8. orientation sensitivity: same scan with the 'ras' layout of the official examples
# Outputs: reports/phase2/
set -euo pipefail
IMG="${1:?usage: run_phase2.sh <mri.nii.gz|dicom_dir|npy> [question]}"
Q="${2:-Can you provide a caption consists of findings for this medical image?}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
[[ -e "$IMG" ]] || { echo "not found: $IMG"; exit 1; }
IMG="$(cd "$(dirname "$IMG")" && pwd)/$(basename "$IMG")"   # absolute, so it survives the cd below
IMG="${IMG#"$PROJECT_DIR"/}"                               # ...and project-relative when inside the project
PY=".venv/bin/python"
OUT="reports/phase2"   # relative: reports never record absolute local paths
CT="upstream/M3D/Data/data/examples/example_03.npy"
SEG_Q="What is liver in this image? Please output the segmentation mask."
export PYTORCH_ENABLE_MPS_FALLBACK=0
cd "$PROJECT_DIR"
mkdir -p "$OUT"

echo "== 1/8 checkpoint verification"; "$PY" scripts/verify_model_dir.py --sha256 --headers | tee "$OUT/verify_model_dir.txt"
echo "== 2/8 preprocessing";           "$PY" scripts/preprocess_mri.py "$IMG" -o "$OUT/input_m3d.npy" --montage "$OUT/input_montage.png" | tee "$OUT/preprocess.txt"
echo "== 3/8 inference + audit";       "$PY" scripts/run_inference.py --image "$IMG" --question "$Q" --dtype bf16 --audit --json-out "$OUT/inference_bf16_audit.json" | tee "$OUT/inference_bf16_audit.txt"
echo "== 4/8 benchmark";               "$PY" scripts/benchmark.py --image "$IMG" --question "$Q" --dtype bf16 --runs 3 --warmup 1 --json-out "$OUT/bench_bf16.json" | tee "$OUT/bench_bf16.txt"
                                       "$PY" scripts/benchmark.py --image "$IMG" --question "$Q" --dtype fp16 --runs 1 --warmup 1 --json-out "$OUT/bench_fp16.json" | tee "$OUT/bench_fp16.txt"
                                       "$PY" scripts/benchmark.py --image "$IMG" --question "$Q" --dtype fp32 --runs 1 --warmup 1 --json-out "$OUT/bench_fp32.json" | tee "$OUT/bench_fp32.txt"
                                       "$PY" scripts/benchmark.py --image "$IMG" --question "$Q" --dtype bf16 --runs 1 --warmup 1 --min-new-tokens 256 --json-out "$OUT/bench_bf16_256tok.json" | tee "$OUT/bench_bf16_256tok.txt"
echo "== 5/8 numerics";                "$PY" scripts/validate_numerics.py --image "$IMG" --question "$Q" --cpu-reference --json-out "$OUT/numerics.json" | tee "$OUT/numerics.txt"
echo "== 6/8 official-code parity"
if [[ -x "$PROJECT_DIR/.venv-ref/bin/python" ]]; then
  "$PY" scripts/compare_upstream.py --image "$IMG" --configs mps-fp32 mps-bf16 mps-fp16 cpu-fp32 --out "$OUT/parity" | tee "$OUT/parity.log"
else
  echo "   skipped: reference env missing -> scripts/setup_reference_env.sh, then re-run" | tee "$OUT/parity.log"
fi
echo "== 7/8 segmentation (official example CT)"
if [[ -f "$CT" ]]; then
  "$PY" scripts/run_inference.py --image "$CT" --task seg --seg --question "$SEG_Q" --dtype bf16 --audit \
      --seg-out "$OUT/ct_seg_liver_mask.npy" --seg-montage "$OUT/ct_seg_liver_montage.png" --json-out "$OUT/ct_seg_liver_bf16.json" | tee "$OUT/ct_seg_liver_bf16.txt"
  "$PY" scripts/benchmark.py --image "$CT" --task seg --seg --question "$SEG_Q" --dtype bf16 --runs 3 --warmup 1 --json-out "$OUT/bench_bf16_seg.json" | tee "$OUT/bench_bf16_seg.txt"
else
  echo "   skipped: $CT missing -> scripts/fetch_upstream.sh"
fi
echo "== 8/8 orientation sensitivity"
if [[ "$IMG" == *.npy ]]; then
  echo "   skipped: $IMG is already preprocessed (orientation applies to NIfTI/DICOM input only)" | tee "$OUT/inference_bf16_ras.txt"
else
  "$PY" scripts/run_inference.py --image "$IMG" --question "$Q" --orientation ras --montage "$OUT/input_montage_ras.png" --json-out "$OUT/inference_bf16_ras.json" | tee "$OUT/inference_bf16_ras.txt"
fi
echo "Phase 2 complete -> $OUT"
