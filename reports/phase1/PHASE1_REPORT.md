# Phase 1 — compatibility assessment (no pretrained weights)

Date: 2026-11-05 · Reproduce: `scripts/run_phase1.sh` (≈ 4 min) · Raw data: `reports/phase1/*.json`

## Environment

| | |
|---|---|
| Machine | Apple M5, 10-core GPU, 32 GB unified memory, macOS 26.0 |
| Python | 3.13.5 arm64 (native, project-local `.venv`) |
| PyTorch | 2.15.0, MPS built + available, Metal recommended working set 24.96 GiB |
| Transformers | 5.22.1 |
| CPU fallback | `PYTORCH_ENABLE_MPS_FALLBACK=0` (unsupported ops raise) |
| CUDA | none: no CUDA wheels, no `.cuda()`/`torch.cuda` anywhere in the port (test-enforced) |

Downloaded: config, tokenizer and original code at the pinned revision (≈ 650 KB), plus the
JSON *headers* of the four safetensors shards via HTTP range requests (≈ 80 KB) to get exact tensor
names, shapes and dtypes. **No weights.** The official GitHub repo was cloned for reference.

## MPS operator probe (fallback disabled)

| Operator (model use) | fp32 | fp16 | bf16 |
|---|---|---|---|
| `avg_pool3d` (projector) | ✅ | ✅ | ✅ |
| SDPA 12×2049×64 (3D ViT) | ✅ | ✅ | ✅ |
| `ConvTranspose3d` k2s2 (SegVol decoder) | ✅ | ❌ not supported | ❌ not supported |
| `LayerNorm` over 4 dims (SegVol decoder) | ✅ | ✅ | ✅ |
| trilinear `interpolate` → 32×256×256 (SegVol output) | ✅ | ✅ | ✅ |
| exact GELU, matmul, `Linear`, `Embedding(padding_idx)` | ✅ | ✅ | ✅ |
| `nonzero`/bool indexing, `multinomial`, `isin`, `sort`+`cumsum` (generate) | ✅ | ✅ | ✅ |

The only gap is replaced by an exact reformulation (`PatchwiseConvTranspose3d`, see PORTING_NOTES #4).

## Tests — 70 passed

| File | What it proves |
|---|---|
| `test_00_environment` (6) | arm64, MPS available, fallback off, no CUDA / MONAI / einops / `trust_remote_code` in the port |
| `test_01_vit3d_monai_parity` (3) | vendored 3D ViT == MONAI ViT (same weights): rel. err < 1e-5, identical keys apart from MONAI ≥ 1.4's unused `norm_cross_attn` |
| `test_02_vision_projector_mps` (8) | **3D vision encoder + projector on MPS with synthetic tensors**: shapes 1×2048×768 → 1×256×3072; vs CPU fp32: fp32 < 1e-4, bf16 < 5e-2, fp16 < 1e-2 rel. err; zero CPU ops; batch 2; SDPA == explicit |
| `test_03_segvol_mps` (8) | ConvTranspose3d replacement exact (< 1e-6); SegVol on MPS fp32/bf16 vs CPU; native fp16/bf16 op still unsupported |
| `test_04_checkpoint_layout` (6) | 617/617 tensor names and shapes == the released shard headers; 4.049 B params; untied `lm_head`; only RoPE buffers non-persistent |
| `test_05_tokenizer` (7) | `M3DTokenizer` == **real Transformers 4.42.4** slow tokenizer on 127 encode + 302 decode goldens; v5 tokenizer agrees on all official templates |
| `test_06_loader` (8) | streaming loader bit-exact on MPS (bf16/fp16/fp32), RoPE fp32, strict failure on missing/unexpected/mis-shaped tensors, sha256 manifest |
| `test_07_tiny_e2e_mps` (11) | multimodal forward/generate on MPS == CPU; KV cache == no cache; eager == sdpa; sampling; `[SEG]` path == manual computation; no-`[SEG]` → −inf; whole generate audited MPS-only |
| `test_08_preprocessing` (7) | orientation phantom (radiological/RAS), invariance to storage order (5 orientations), MONAI CropForeground+Resize parity < 1e-5, example pass-through |
| `test_09_scripts` (6) | Phase-2 CLIs (`run_inference`, `benchmark`, `validate_numerics`, `preprocess_mri`, `verify_model_dir`, `check_env`) rehearsed end-to-end on a tiny synthetic checkpoint |

## Benchmarks (synthetic weights; compute and memory identical to the real checkpoint)

**Vision encoder + projector** (1×1×32×256×256 → 1×256×3072) and **SegVol**, batch 1, median of 5:

| | fp32 | bf16 | fp16 |
|---|---|---|---|
| 3D ViT + projector | 165 ms | 79 ms | 80 ms |
| SegVol (encoder + decoder + upsampling) | 180 ms | 88 ms | 88 ms |

**Full model** (vision + 272-token prompt prefill + greedy decode, decode length forced):

| dtype | weights | TTFT (vision part) | decode | 256 new tokens | peak MPS alloc | peak MPS driver | peak process footprint |
|---|---|---|---|---|---|---|---|
| bf16 | 7.54 GiB | 0.38 s (0.08 s) | 14.5 tok/s | 17.9 s | 7.77 GiB | 8.37 GiB | 8.70 GiB |
| fp16 | 7.54 GiB | 0.37 s (0.08 s) | 14.7 tok/s | 17.8 s | 7.77 GiB | 8.37 GiB | 8.70 GiB |
| fp32 | 15.08 GiB | 0.91 s (0.17 s) | 7.6 tok/s | (128 tok: 17.6 s) | 15.41 GiB | 16.01 GiB | 16.34 GiB |

bf16 + SegVol + op-level device audit (32 tokens): **69,401 ATen compute ops, all on MPS, 0 on CPU**;
host↔device traffic was the input upload plus one 1-element bookkeeping tensor per decode step
from HF `generate`.

Interpretation: decode reads the whole 7.5 GiB of weights per token, so 14.5 tok/s ≈ 117 GB/s,
about 75 % of the M5's 153 GB/s memory bandwidth. All dtypes leave ≥ 15 GB of the 32 GB free.

## Phase-1 verdict

**Pass.** The full inference graph (3D ViT, projector, Phi-3 decoder with KV cache, SegVol) runs
natively on MPS in bf16/fp16/fp32, with no CUDA dependencies and no CPU fallback. It is
checkpoint-compatible down to every tensor name and shape. Ready for Phase 2 (pretrained weights).
