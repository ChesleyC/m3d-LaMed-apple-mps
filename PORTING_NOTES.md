# Porting notes — M3D-LaMed-Phi-3-4B → Apple Silicon (PyTorch MPS)

**Upstream, pinned**

| Source | Revision |
|---|---|
| Model: `GoodBaiBai88/M3D-LaMed-Phi-3-4B` (Apache-2.0) | `329bed510c2cff0ad3fe6f2a1a7075f45e2706e9` (2024-07-04) |
| Code: `BAAI-DCAI/M3D` | `63e72005a2304eda74579870239bfc7b04b0edae` (2025-04-13) |
| Upstream software stack | torch 2.2.1+cu118, transformers 4.39–4.42, MONAI 1.3.0 |
| This port | torch 2.15.0 (MPS), transformers 5.22.1, Python 3.13 arm64, macOS 26.0, Apple M5 / 32 GB |

The original remote code (`modeling_m3d_lamed.py`) is kept verbatim in
`upstream/hf_M3D-LaMed-Phi-3-4B/` for reference. **The port never imports or executes it.** The port lives
in `m3d_lamed_mps/` and loads the original safetensors 1:1. The one exception is verification only:
`scripts/upstream_reference.py` runs the official code unchanged in a separate environment `.venv-ref` with
the original stack (torch 2.2.1, transformers 4.42.4, MONAI 1.3.0) to produce reference outputs. The port
and the reference never import each other (test-enforced).

## Model recap

```
MRI 1×32×256×256 ─► 3D ViT (87.4 M, patch 4×16×16 → 2048 tokens, +CLS) ─► SpatialPoolingProjector
                     (2×2×2 avg-pool → 256 tokens, MLP 768→3072→3072, 11.8 M)
prompt  <s> + <im_patch>×256 + question ─► Phi-3-mini embeddings; positions 1..256 ← image tokens
                     ─► Phi-3-mini decoder (32 layers, 3072, 3.82 B) ─► text
[SEG] hidden state ─► seg_projector (11.8 M) ─► SegVol (ViT encoder + SAM-3D decoder, 117.3 M) ─► mask 1×32×256×256
```
617 tensors, 4.049 B parameters, stored fp32 (16.2 GB).

## Modifications

| # | Upstream | Problem on MPS / current stack | Change in this port | Verified by |
|---|---|---|---|---|
| 1 | `torch.zeros(...).cuda()` building `seg_token_mask` (training forward with `segs`) | crashes off-CUDA | training-only seg-loss branch removed; inference seg path builds every tensor on the input's device | `test_00_environment::test_no_cuda_in_port_sources` |
| 2 | vision tower and SegVol encoder use MONAI `ViT`/`PatchEmbeddingBlock`/`TransformerBlock` (1.3.0 API) | `pos_embed=` arg removed in MONAI 1.4; MONAI ≥ 1.4 adds `norm_cross_attn` parameters to every block (24 tensors not in the checkpoint) | `vit3d.py`: pure-PyTorch re-implementation of the MONAI 1.3.0 math with identical parameter names; MONAI no longer a runtime dependency | `test_01`: rel. err < 1e-5 vs MONAI 1.6.1 (same weights), identical key sets |
| 3 | einops `Rearrange` in patch embedding / projector | — (dependency only) | `view`/`permute`/`flatten` | `test_01`, `test_02`: bit-exact vs einops |
| 4 | `nn.ConvTranspose3d(k=2, s=2)` ×2 in the SegVol mask decoder | PyTorch 2.15: *"ConvTranspose 3D with BF16 or FP16 types is not supported on MPS"* | `PatchwiseConvTranspose3d`: exact `matmul + reshape` formulation for `kernel == stride` (same parameters, all dtypes on MPS) | `test_03`: rel. err < 1e-6 vs native; a test asserts the native op still fails (signals when the workaround can go) |
| 5 | attention in ViT/SAM via `einsum` + `softmax` | slow, memory-heavy on MPS | `F.scaled_dot_product_attention` (fused Metal kernel); `set_vision_sdpa(False)` restores the explicit path | `test_02/03`: SDPA vs explicit rel. err < 1e-4 |
| 6 | `LamedPhi3Config(Phi3Config)` relied on loose attributes | Transformers 5 configs are strict dataclasses | multimodal fields declared explicitly (defaults = release) | `test_04::test_config_semantics` |
| 7 | `forward`/`generate`/`prepare_inputs_for_generation` written against Transformers 4.42 | signature and cache changes in v5 | `forward(input_ids, images, **kwargs)` passthrough; `generate(images, input_ids, seg_enable, **gen_kwargs)` keeps the upstream call convention and return values; no `prepare_inputs_for_generation` override | `test_07`: CPU==MPS greedy tokens, KV cache == no cache, sdpa == eager |
| 8 | `[SEG]` prompt = last hidden state at `output_ids[:,1:]==seg_id` | v5 collects hidden states via hooks | verified that `hidden_states[step][-1]` is the post-norm state, same as 4.42 (max diff 2.6e-6); selection logic unchanged | `test_07::test_seg_path_matches_manual_computation` |
| 9 | `AutoTokenizer(..., use_fast=False, trust_remote_code=True)` → Transformers-4.42 slow sentencepiece tokenizer | Transformers 5 removed it; its `LlamaTokenizer` differs from 4.42 on 10/127 test inputs (repeated whitespace, newlines/tabs, leading space, literal `▁`) and decodes `[SEG]`/`<\|end\|>` with different spacing | `tokenizer.py` (`M3DTokenizer`): exact port of the 4.42.4 slow algorithm on sentencepiece | `test_05`: 127 encode + 302 decode goldens from **real Transformers 4.42.4** — 0 mismatches |
| 10 | `AutoModelForCausalLM.from_pretrained(trust_remote_code=True, device_map='auto')` | executes downloaded code; `device_map='auto'` is meaningless on unified memory | `loading.py`: model built on `meta`, cast, materialised directly on MPS, every safetensors tensor streamed in; fails on any missing/unexpected/mis-shaped tensor | `test_04`: 617/617 names+shapes vs shard headers; `test_06`: bit-exact streaming, failure modes |
| 11 | RoPE `inv_freq` (non-persistent buffer) | `to_empty()` leaves it uninitialised; a blanket `.to(bf16)` would also reduce RoPE precision | rebuilt in float32 after materialisation | `test_06::test_streaming_load_to_mps_is_exact` |
| 12 | `prepare_inputs_for_multimodal` overwrote positions 1..256 unconditionally | a malformed prompt silently overwrote question tokens | raises unless `input_ids[:,1:257]` are all `<im_patch>` | `test_07::test_misplaced_image_tokens_raise` |
| 13 | small upstream bugs | `ViT3DTower` indexed hidden states with a *string* for `select_layer<-1`; `.dtype`/`.device` properties referenced attributes `nn.Module` lacks; `generate` dropped `attention_mask`; seg path crashed if only one token was generated | fixed (no effect on the released config, `select_layer=-1`) | code review, `test_07` |
| 14 | dead/training code: 2D SAM `ImageEncoderViT`, window attention, `Sam` wrapper, Dice/BCE losses, `initialize_*` training helpers, debug `print`s | — | removed; mask prompts in `PromptEncoder` raise `NotImplementedError` (never used by M3D) | `test_04` (no checkpoint tensor lost) |
| 15 | preprocessing for Radiopaedia PNG stacks only | MRI arrives as NIfTI/DICOM in scanner space | `preprocessing.py`: SimpleITK reader, reorientation to an axial stack, percentile clip, then exactly MONAI `CropForeground()` + `Resize([32,256,256], "bilinear")` | `test_08`: MONAI parity < 1e-5, orientation phantom, storage-order invariance |
| 16 | CPU fallback could hide missing kernels | — | scripts force `PYTORCH_ENABLE_MPS_FALLBACK=0`; `require_mps()` refuses to run otherwise; `DeviceAudit` (TorchDispatchMode) records the device of every ATen op | full-size bf16 run with SegVol: 69,401 ops on MPS, 0 on CPU |

Phase 2 (original weights) needed **no change to the model code**: the checkpoint ran on the Phase-1
implementation as is and reproduced the official outputs (below).

Precision: bf16 by default (as the official Quickstart); fp16 and fp32 are also supported. Note that
the official benchmark scripts (`Bench/eval/eval_caption.py`) loaded the model **in fp32** (no
`torch_dtype`), which fits on this machine too (peak 16.3 GiB) and is used as the Phase-2 reference.

## Phase-1 verification (no pretrained weights)

Everything below runs with synthetic weights/tensors and `PYTORCH_ENABLE_MPS_FALLBACK=0`;
`scripts/run_phase1.sh` reproduces it (70 tests + benchmarks, ~4 min). Details: `reports/phase1/PHASE1_REPORT.md`.

* MPS op probe: every operator of the model has a native MPS kernel in fp32/fp16/bf16 **except**
  `ConvTranspose3d` in fp16/bf16 (→ modification 4).
* Vision encoder + projector on MPS vs CPU fp32: fp32 rel. err < 1e-4, bf16 < 5e-2, fp16 < 1e-2.
* Full-size model (random weights, identical compute/memory): bf16 TTFT 0.38 s, 14.5 tok/s decode,
  256 tokens in 17.9 s, peak 7.8 GiB MPS / 8.7 GiB process footprint.

## Phase-2 verification (original weights, official example CTs)

`scripts/run_phase2.sh upstream/M3D/Data/data/examples/example_05.npy` reproduces all of it (8 steps,
≈ 6 min) on public data. Details: `reports/phase2/PHASE2_REPORT.md`.

* **Checkpoint:** sha256 of all 11 required files, and 617/617 tensor names and shapes, match the pinned revision.
* **Parity with the official implementation:** the official remote code ran unchanged (CPU fp32,
  original stack) on 4 cases from the official example CTs (caption, organ VQA, second caption, liver
  segmentation). The port on MPS (fp32, fp16, bf16) and on CPU (fp32) reproduces every case:
  * identical input ids and identical greedy output tokens;
  * 100 % teacher-forced argmax agreement;
  * logits relative error ≈ 1e-5 in fp32, ≤ 5e-3 in fp16, ≤ 4e-2 in bf16;
  * liver mask Dice 1.0000 / 0.9999 / 0.9989.
* **GPU execution:** op-level audit with fallback disabled. Caption: 47,352 ATen ops on MPS.
  Segmentation incl. SegVol: 22,288. **0 CPU compute ops** in both.
* **Latency and memory, bf16:**
  * time to first token 0.38 s, of which 3D ViT + projector 0.08 s;
  * decode 14.6 tok/s; 256 tokens take 18.1 s;
  * peak 7.7 GiB MPS, 8.7 GiB process footprint (fp32: 0.91 s / 7.6 tok/s / 16.4 GiB).

## Remaining limitations / open questions

1. **Scope of the equivalence proof.** Token-level identity with the official code is shown for greedy
   decoding on 4 cases (2 official example CTs, incl. segmentation), plus a tiny-checkpoint test in the
   suite. Sampling (`do_sample=True`, Quickstart `top_p=0.9`) is stochastic, and its random streams differ
   between devices, so sampled texts are not comparable token by token. Their distributions are the same
   because the logits match.
2. **Domain shift.** M3D-Cap was built from Radiopaedia (folders `ct_case`, `ct_quizze`), and the
   training data is CT-dominated. Output for other modalities, MRI in particular, can be fluent
   radiology prose describing findings or organs that are not there. Validate on your own data. This is
   a property of the released model, not of the port. It is a research model and is not clinically validated.
3. **Orientation is ambiguous in the training data.** The official example volumes (M3D-Seg) are
   RAS-indexed (rows posterior→anterior, columns left→right, slices inferior→superior), whereas
   Radiopaedia images (M3D-Cap, the captioning data) are in radiological display orientation. The default
   for NIfTI/DICOM input is `radiological`; `--orientation ras` reproduces the example layout, and
   `--reverse-slices` flips the slice order (unknown for M3D-Cap). The output can change substantially
   with the layout. Protocol step 8 runs both layouts for NIfTI/DICOM input.
4. **MRI intensity normalisation is a heuristic.** The training JPEGs were display-windowed 8-bit
   images; the default 0.5/99.5-percentile clip stands in for that windowing (`--no-clip` gives
   plain min-max).
5. **Resampling ignores voxel spacing and has no anti-aliasing,** exactly like the official pipeline;
   a 1 mm brain volume is reduced to 32 axial slices.
6. **Batching:** prompts of different lengths are not supported (no padding), as upstream (whose
   generate also ignored the attention mask). Batch = 1 is the supported mode.
7. **Numerics differ from upstream at rounding level:** SDPA vs einsum attention and the
   ConvTranspose3d reformulation agree to ≤ 1e-5 relative error in fp32. With the original weights, the
   logits relative error vs the official fp32 code is ≈ 1e-5 (fp32), ≤ 5e-3 (fp16) and ≤ 4e-2 (bf16),
   without changing any greedy token in the tested cases. fp16 is the more accurate half-precision mode
   at the same speed; bf16 stays the default as in the official Quickstart.
8. **Memory measurement:** PyTorch has no MPS `max_memory_allocated`; peaks come from a 2 ms
   allocator sampler plus the macOS per-process `lifetime_max_phys_footprint`, which counts GPU
   allocations on unified memory.
9. **Host↔device traffic:** besides input upload and output download, Hugging Face `generate`
   moves one 1-element bookkeeping tensor per decode step. This is not CPU compute.
10. **Throughput ceiling:** bf16 decode is about 75 % of the 153 GB/s memory-bandwidth roofline
    (7.5 GiB read per token). Going faster would need weight quantization, which changes the original
    checkpoint's numerics, so it is deliberately not done.
11. **Not ported:** training/fine-tuning, LoRA merging, bitsandbytes, the Gradio online demo,
    GGUF/MLX conversions.
12. **The reference environment is for verification only.** `.venv-ref` executes the downloaded official
    code (`trust_remote_code`) on CPU. It needs Python 3.10–3.12 (no torch 2.2 wheels for 3.13) and about
    0.8 GB, and it is optional: without it, protocol step 6 and `test_10` are skipped.
