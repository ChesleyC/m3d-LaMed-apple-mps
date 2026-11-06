#!/usr/bin/env bash
# Phase 1 (no pretrained weights needed): environment check, full test suite, synthetic benchmarks.
# Outputs: reports/phase1/*.json, reports/phase1/pytest.txt
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY=".venv/bin/python"
OUT="reports/phase1"   # relative: reports never record absolute local paths
export PYTORCH_ENABLE_MPS_FALLBACK=0
cd "$PROJECT_DIR"
mkdir -p "$OUT"

"$PY" scripts/check_env.py | tee "$OUT/check_env.txt"
"$PY" -m pytest -q 2>&1 | tee "$OUT/pytest.txt"

for comp in vision segvol; do
  for dt in fp32 bf16 fp16; do
    "$PY" scripts/benchmark.py --component "$comp" --dtype "$dt" --runs 5 --warmup 2 --json-out "$OUT/bench_${comp}_${dt}.json" >/dev/null
  done
done
"$PY" scripts/benchmark.py --synthetic --dtype bf16 --max-new-tokens 256 --runs 3 --warmup 1 --json-out "$OUT/bench_full_synthetic_bf16.json"
"$PY" scripts/benchmark.py --synthetic --dtype fp16 --max-new-tokens 256 --runs 2 --warmup 1 --json-out "$OUT/bench_full_synthetic_fp16.json"
"$PY" scripts/benchmark.py --synthetic --dtype fp32 --max-new-tokens 128 --runs 1 --warmup 1 --json-out "$OUT/bench_full_synthetic_fp32.json"
"$PY" scripts/benchmark.py --synthetic --dtype bf16 --max-new-tokens 32 --runs 1 --warmup 1 --seg --audit --json-out "$OUT/bench_full_synthetic_bf16_seg_audit.json"
echo "Phase 1 complete -> $OUT"
