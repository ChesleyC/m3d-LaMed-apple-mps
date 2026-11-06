# Phase 2: original pretrained weights

Date: 2026-11-05 · Machine: Apple M5 (10-core GPU), 32 GB, macOS 26.0 · PyTorch 2.15.0 (MPS), Transformers 5.22.1

Reproduce: `scripts/fetch_upstream.sh && scripts/setup_reference_env.sh && scripts/run_phase2.sh upstream/M3D/Data/data/examples/example_05.npy`
(8 steps, ≈ 6 min). Raw data: `reports/phase2/*.json`, `*.txt`, `*.log`.

**Verdict: pass.**
* The original checkpoint loads unchanged and runs entirely on the GPU, with no CPU fallback and no
  CPU compute ops.
* On the same inputs, the port reproduces the **official implementation** token for token in fp32,
  fp16 and bf16. The official code ran unchanged with its original software stack. The SegVol masks
  match too (Dice 0.9989 to 1.0000).
* bf16 answers start after 0.38 s and decode at 14.6 tok/s, with a peak process footprint of 8.7 GiB.
* **No model-code change was needed** on top of Phase 1.

All inputs are public: the example CT volumes shipped with the official
[BAAI-DCAI/M3D](https://github.com/BAAI-DCAI/M3D) repository (`Data/data/examples/`, already
preprocessed to 1×32×256×256). Anyone can rerun the protocol and get the same numbers (timings vary by machine).

## Inputs

* **Checkpoint** `models/` (GoodBaiBai88/M3D-LaMed-Phi-3-4B @ 329bed5):
  * all 11 required files match the pinned manifest's sha256;
  * all 617 tensors match the shard headers by name and shape (`verify_model_dir.txt`).
* **Main input** `example_05.npy`: abdomen–pelvis CT, used for the caption, VQA, benchmark and numerics steps.
* **Segmentation input** `example_03.npy`: abdomen–thorax CT, with the liver-segmentation prompt from
  the official README.

## 1. The output is the official model's output

`scripts/compare_upstream.py` runs the **verbatim official remote code** as the reference: 
`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` plus the slow `AutoTokenizer`, as
in the official Quickstart. It runs in an isolated environment with the stack the checkpoint was built
with: Python 3.11, torch 2.2.1, transformers 4.42.4, MONAI 1.3.0. It runs on CPU in fp32, which is how
the official benchmark evaluated. The port then runs the same inputs and prompts.

Four cases, greedy decoding, ≤ 256 tokens:
* `example_05` with the caption prompt and an organ VQA question;
* `example_03` with the caption prompt and the liver-segmentation prompt.

| Port config | input ids | image tokens cos | teacher-forced logits: argmax agreement / cos (min) / rel. err | free-running greedy output | liver mask Dice vs official |
|---|---|---|---|---|---|
| MPS fp32 | 4/4 identical | 1.000000 | 100 % / 1.000000 (1.00000) / 3e-6 – 1e-5 | **identical, 4/4** | 1.0000 |
| MPS fp16 | 4/4 identical | 1.000000 | 100 % / ≥ 0.999987 (0.99996) / 1.3e-3 – 4.6e-3 | **identical, 4/4** | 0.9999 |
| MPS bf16 | 4/4 identical | 0.99999 | 100 % / ≥ 0.99909 (0.99643) / 1.1e-2 – 3.8e-2 | **identical, 4/4** | 0.9989 (62,131 vs 62,119 voxels) |
| CPU fp32 | 4/4 identical | 1.000000 | 100 % / 1.000000 (1.00000) / 2e-6 – 1e-5 | **identical, 4/4** | 1.0000 |

How to read the table:
* **Teacher-forced logits:** both models score the same token sequence (the official output), so
  every generated position is compared.
* **Port vs official:** port CPU fp32 ≈ port MPS fp32 ≈ official, to about 1e-5 (SDPA vs eager
  attention). So the implementation and the Metal kernels both reproduce the official computation;
  the bf16/fp16 differences are ordinary rounding.
* **Decoding:** the port decodes the official token ids to exactly the same strings as the 4.42 tokenizer.

Official (CPU fp32) outputs, reproduced exactly by every port configuration (`parity/parity.json`):

| Case | Output |
|---|---|
| `example_05` caption | "No acute fracture.Normal pelvic CT in an older adult female uploaded for reference." |
| `example_05` organ VQA | "Kidney" |
| `example_03` caption | "Heterogeneously enhancing mass in the upper pole of the left kidney.Bony enlargement and coarse trabeculation in the right hemipelvis are seen on CT." |
| `example_03` liver segmentation | "The segmentation reveals  [SEG] ." + a mask of 62,119 voxels covering the liver from its tip to the dome (overlay montage written locally to `ct_seg_liver_montage.png` by step 7) |

The test suite runs the same comparison on a tiny random checkpoint (`tests/test_10_upstream_parity.py`).
One variant is biased to emit `[SEG]`, so the official SegVol path is also compared on a non-empty mask.

### Numerics vs a CPU-fp32 port run (`numerics.json`, `example_05`, 64 greedy tokens)

| | image tokens cos | next-token logits cos / rel. err | top-1 / top-5 | greedy output |
|---|---|---|---|---|
| MPS fp32 | 1.000000 | 1.000000 / < 5e-5 | = / 5 of 5 | identical |
| MPS fp16 | 1.000000 | 0.999996 / 0.0042 | = / 5 of 5 | identical |
| MPS bf16 | 0.999988 | 0.999876 / 0.0163 | = / 4 of 5 | identical |

fp16 is about 4–8× closer to fp32 than bf16 and just as fast. bf16 stays the default, as in the
official Quickstart.

## 2. GPU execution

The op-level `DeviceAudit` (TorchDispatchMode) records the device of every ATen op, with
`PYTORCH_ENABLE_MPS_FALLBACK=0`:

| Run (bf16, original weights) | ATen compute ops on MPS | CPU compute ops | host↔device copies |
|---|---|---|---|
| `example_05` caption, 21 tokens (`inference_bf16_audit.json`) | 47,352 | **0** | 54 (32 `to`, 22 `copy_`) |
| `example_03` liver segmentation incl. SegVol (`ct_seg_liver_bf16.json`) | 22,288 | **0** | 42 (32 `to`, 10 `copy_`) |

The copies are:
* the input upload;
* the output download;
* Hugging Face `generate`'s per-step scalar bookkeeping (the matching CPU-side scalar ops
  `lift_fresh`/`detach_`/`is_nonzero` are listed in the JSON).

None of them is CPU compute.

## 3. Latency and peak memory (original weights)

`example_05`, 272-token prompt, greedy, batch 1, median of runs after 1 warmup:

| Config | time to first token | of which 3D ViT + projector | decode | total | peak MPS allocated | peak MPS driver | peak process footprint |
|---|---|---|---|---|---|---|---|
| bf16, 21-token answer (3 runs) | 0.376 s | 0.080 s | 14.6 tok/s | 1.82 s | 7.67 GiB | 8.36 GiB | 8.73 GiB |
| fp16, 21-token answer | 0.373 s | 0.080 s | 14.5 tok/s | 1.82 s | 7.67 GiB | 8.36 GiB | 8.72 GiB |
| fp32, 21-token answer | 0.907 s | 0.168 s | 7.6 tok/s | 3.66 s | 15.32 GiB | 16.01 GiB | 16.37 GiB |
| bf16, 256 tokens (forced) | 0.372 s | 0.080 s | 14.4 tok/s | 18.08 s | 7.77 GiB | 8.37 GiB | 8.73 GiB |
| bf16, `example_03` liver segmentation, 9 tokens + mask (3 runs) | 0.374 s | 0.080 s | 14.4 tok/s | 1.09 s | 7.79 GiB | 8.37 GiB | 9.04 GiB |

* **Segmentation:** SegVol after `[SEG]` (encoder over the full 32×256×256 volume plus the mask
  decoder) takes **0.09 s** (`bench_bf16_seg.json`).
* **Checkpoint load:** fp32 safetensors → bf16 on MPS, 617 tensors streamed: 9–16 s (warm vs cold page cache).
* **Phase-1 predictions held:** decode speed and memory match the synthetic-weight measurements of
  Phase 1 (bf16 14.5 tok/s, 7.77 GiB). The 32 GB machine keeps ≥ 15 GB free even in fp32.

## 4. Issues found and fixed in Phase 2

| Issue | Fix |
|---|---|
| `scripts/run_phase1.sh` and `run_phase2.sh` lacked the executable bit (*permission denied*) | `chmod +x`; test `test_shell_scripts_are_executable` |
| `validate_numerics` printed cosine 1.000024 (float32 accumulation) | cosines and relative errors computed in float64 |
| Reports recorded absolute local paths; `run_phase2.sh` broke when given a relative path from another directory | paths are written relative to the project (`display_path`); the protocol scripts resolve their input before changing directory |
| No independent check against the *official* code with real weights | `setup_reference_env.sh` + `upstream_reference.py` (official code, original stack) + `compare_upstream.py`; protocol step 6; `test_10` |
| No way to run many scans or inspect masks | `batch_inference.py` (one load, resumable); `--seg-montage` (red overlay); `benchmark.py --task seg` with segmentation timing |

The runtime package `m3d_lamed_mps/` is unchanged from Phase 1, except that `save_montage` gained an
optional mask overlay and `loading.py` a path-display helper.

## 5. Remaining limitations

See [PORTING_NOTES.md](../../PORTING_NOTES.md#remaining-limitations--open-questions). In short:

* **Token-level identity is shown for greedy decoding on four cases.** Sampling is stochastic, and its
  random streams differ between devices.
* **The model was trained mostly on CT.** Validate its output on your own data, especially for MRI.
* **Batch size 1 only,** as upstream.
