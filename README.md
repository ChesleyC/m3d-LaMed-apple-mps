# m3d-LaMed-apple-mps

Inference for **M3D-LaMed-Phi-3-4B**, a 3D medical vision-language model (3D ViT + Phi-3-mini +
SegVol) from [BAAI-DCAI/M3D](https://github.com/BAAI-DCAI/M3D), ported to **Apple Silicon / PyTorch MPS**.
It loads the original checkpoint [`GoodBaiBai88/M3D-LaMed-Phi-3-4B`](https://huggingface.co/GoodBaiBai88/M3D-LaMed-Phi-3-4B)
unchanged (no retraining, no conversion) and runs 3D-image-to-text and segmentation entirely on the GPU.

**Author:** Cheng Cheng (Benjamin Cheng), Stanford University · [ChesleyC](https://github.com/ChesleyC) ·
[OpenReview](https://openreview.net/profile?id=~Cheng_Cheng18)

> Unofficial port, not affiliated with BAAI or the M3D authors. Research use only; not for clinical use.

* no CUDA, no CPU fallback (`PYTORCH_ENABLE_MPS_FALLBACK=0` is enforced), no `trust_remote_code`
* no MONAI/einops at runtime; Transformers 5 + PyTorch 2.15
* checkpoint-compatible down to all 617 tensor names/shapes; tokenizer identical to the original
  (verified against Transformers 4.42.4)
* **verified against the official implementation**: with the original weights, the port on MPS
  reproduces the official code token for token, in fp32, fp16 and bf16, including the SegVol mask
  (Dice ≥ 0.998). The official code ran unchanged with torch 2.2.1 / transformers 4.42.4 / MONAI 1.3.0.

| Status | |
|---|---|
| **Phase 1** — compatibility (synthetic tensors, no weights) | ☑ done, [report](reports/phase1/PHASE1_REPORT.md) |
| **Phase 2** — original weights, official example CTs | ☑ done, [report](reports/phase2/PHASE2_REPORT.md) |

All changes vs upstream and the known limitations: **[PORTING_NOTES.md](PORTING_NOTES.md)**.

## Requirements

* Apple Silicon Mac (developed on an M5 with 32 GB), macOS ≥ 15, native arm64 Python ≥ 3.11
* about 17 GB of free disk for the weights
* memory (peak process footprint): bf16/fp16 ≈ 9 GiB, fp32 ≈ 16.4 GiB

## Setup

```bash
scripts/setup_env.sh          # creates .venv (pinned: requirements-lock.txt); or PYTHON=/path/to/python3.13 scripts/setup_env.sh
scripts/fetch_upstream.sh     # clones the official M3D repo (example CT volumes) into upstream/M3D (git-ignored)
.venv/bin/python scripts/check_env.py
```

`upstream/hf_M3D-LaMed-Phi-3-4B/` (pinned config, tokenizer, original code and tensor headers, no
weights) is part of this repository. The official example CTs are only needed for the Phase-2
protocol and one test; without them those parts are skipped.

## Phase 1 (no weights)

```bash
scripts/run_phase1.sh        # check_env + pytest + synthetic benchmarks -> reports/phase1/
```

## Phase 2 (original weights)

1. **Download the checkpoint** (~16.2 GB, pinned revision) into `models/` and verify it:
   ```bash
   scripts/download_weights.sh --dry-run    # list only
   scripts/download_weights.sh              # download + sha256 + tensor-header verification
   ```
   Placing the files in `models/` by hand works too (`model-0000{1..4}-of-00004.safetensors`,
   `model.safetensors.index.json`, `config.json`, `generation_config.json`, `tokenizer.model`,
   `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`). Then run
   `.venv/bin/python scripts/verify_model_dir.py --sha256 --headers`. You don't need `model.bin` or
   `mm_projector.bin` from the HF repo.
2. **Optional, recommended once:** the comparison with the official implementation (protocol step 6)
   needs a separate reference environment with the original software stack (Python 3.11 from
   conda-forge, torch 2.2.1, transformers 4.42.4, MONAI 1.3.0; ≈ 0.8 GB, git-ignored):
   ```bash
   scripts/setup_reference_env.sh
   ```
3. **Run the full protocol.** To reproduce the published report, use the official example CT:
   ```bash
   scripts/run_phase2.sh upstream/M3D/Data/data/examples/example_05.npy
   ```
   Your own scan also works (NIfTI, a DICOM directory, or a preprocessed 1×32×256×256 `.npy`):
   `scripts/run_phase2.sh /path/to/scan.nii.gz`.

   The protocol has 8 steps (≈ 6 min). Results go to `reports/phase2/`:
   1. verify the checkpoint;
   2. preprocess the input, with a PNG montage;
   3. image-to-text with the op-level device audit;
   4. latency and peak memory (bf16/fp16/fp32, 256 tokens);
   5. bf16/fp16 numerics against fp32 references;
   6. comparison with the official code;
   7. liver segmentation on an official example CT;
   8. orientation sensitivity (NIfTI/DICOM input only).
4. **Many scans at once** (one model load, resumable JSONL + summary statistics):
   ```bash
   .venv/bin/python scripts/batch_inference.py /path/to/scans/ --jsonl-out /path/to/out/captions.jsonl
   ```

Single runs:

```bash
.venv/bin/python scripts/run_inference.py --image scan.nii.gz                      # default caption prompt, bf16, greedy
.venv/bin/python scripts/run_inference.py --image scan.nii.gz \
    --question "Describe the findings of the medical image you see." --audit --json-out out.json
.venv/bin/python scripts/run_inference.py --image ct.nii.gz --task seg --seg \
    --question "What is liver in this image? Please output the segmentation mask." --seg-out mask.npy --seg-montage mask.png
.venv/bin/python scripts/preprocess_mri.py scan.nii.gz -o scan_m3d.npy --montage scan_m3d.png
```

Python API:

```python
import os; os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import torch
from m3d_lamed_mps.pipeline import M3DLaMedMPS
from m3d_lamed_mps.preprocessing import preprocess_volume

pipe = M3DLaMedMPS.from_pretrained("models", dtype=torch.bfloat16)          # original weights on MPS
image, info = preprocess_volume("scan.nii.gz")                               # 1x32x256x256 in [0,1]
res = pipe.generate(image, question="Can you provide a caption consists of findings for this medical image?")
print(res.text, res.timings, res.memory)
# low level, same call convention as upstream:  pipe.model.generate(image_t, input_ids, seg_enable=False, max_new_tokens=256)
```

## Performance (M5, 32 GB, original weights, official example CT)

272-token prompt, greedy, batch 1, median of runs ([raw data](reports/phase2/)):

| dtype | time to first token (3D ViT + projector) | decode | 21-token answer | 256 tokens | peak MPS alloc | peak process footprint |
|---|---|---|---|---|---|---|
| bf16 | 0.38 s (0.08 s) | 14.6 tok/s | 1.8 s | 18.1 s | 7.7 GiB | 8.7 GiB |
| fp16 | 0.37 s (0.08 s) | 14.5 tok/s | 1.8 s | — | 7.7 GiB | 8.7 GiB |
| fp32 | 0.91 s (0.17 s) | 7.6 tok/s | 3.7 s | — | 15.3 GiB | 16.4 GiB |

Other timings:
* loading the checkpoint (16.2 GB fp32 safetensors → bf16 on MPS) takes 9–16 s;
* the SegVol mask after `[SEG]` takes ≈ 0.09 s, and a complete segmentation request 1.1 s.

**About the model itself.** M3D-LaMed was trained mostly on CT (M3D-Cap was built from Radiopaedia).
Output for other modalities, MRI in particular, can be fluent but wrong. Validate it on your own data
before relying on it.

## Layout

```
m3d_lamed_mps/            the port (runtime package)
  modeling_m3d_lamed.py     LamedPhi3ForCausalLM: vision tower, projector, multimodal generate, [SEG] path
  vit3d.py                  MONAI-1.3.0-equivalent 3D ViT (pure PyTorch)
  segvol.py                 SegVol / SAM-3D decoder (MPS-native ConvTranspose3d)
  configuration_m3d_lamed.py, tokenizer.py (exact 4.42 slow tokenizer), loading.py (strict streaming loader)
  preprocessing.py          NIfTI/DICOM -> 1x32x256x256 (official CropForeground + Resize)
  pipeline.py               timed image-to-text wrapper; mps_utils.py: no-fallback guard, op audit, memory
  prompts.py                official prompt templates; checkpoint_manifest.json: sizes + sha256 (pinned)
scripts/                  setup, fetch, download, verify, preprocess, run_inference, batch_inference, benchmark,
                          validate_numerics, compare_upstream (+ upstream_reference, setup_reference_env), run_phase{1,2}
tests/                    75 tests (pytest); test_10 runs the official code and needs .venv-ref, otherwise it is skipped
upstream/                 pinned reference: HF config/tokenizer/original code + shard headers (M3D clone: git-ignored)
models/                   checkpoint goes here (git-ignored)
reports/                  phase reports and raw JSON
LICENSE                   MIT licence of this project
third_party_licenses/     full Apache-2.0 text, required for the adapted third-party parts
THIRD_PARTY_NOTICES.md    credits and licences of upstream code and files
CITATION.cff              citation metadata ("Cite this repository")
```

## Citation

If you use this port, please cite it (GitHub's **"Cite this repository"** button reads
[CITATION.cff](CITATION.cff)):

```bibtex
@software{cheng2026m3dlamedmps,
  author  = {Cheng, Cheng},
  title   = {m3d-LaMed-apple-mps: M3D-LaMed-Phi-3-4B on Apple Silicon (PyTorch MPS)},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/ChesleyC/m3d-LaMed-apple-mps}
}
```

It builds on the work of the M3D and SegVol authors. Please also cite their papers:

```bibtex
@misc{bai2024m3d,
  title={M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models},
  author={Fan Bai and Yuxin Du and Tiejun Huang and Max Q. -H. Meng and Bo Zhao},
  year={2024}, eprint={2404.00578}, archivePrefix={arXiv}, primaryClass={cs.CV}
}
@article{du2023segvol,
  title={SegVol: Universal and Interactive Volumetric Medical Image Segmentation},
  author={Du, Yuxin and Bai, Fan and Huang, Tiejun and Zhao, Bo},
  journal={arXiv preprint arXiv:2311.13385}, year={2023}
}
```

## License

[MIT](LICENSE), Copyright (c) 2026 Cheng Cheng (Benjamin Cheng). If you reuse this code, keep the
copyright notice and the licence text.

Parts are adapted from, or include files of, BAAI-DCAI/M3D and SegVol (MIT), the M3D-LaMed model files,
MONAI, Segment Anything and Hugging Face Transformers (Apache-2.0), and the Phi-3 tokenizer (MIT). Those
licences continue to apply to their parts; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The
model weights are not included. They are downloaded from Hugging Face under their own licence (Apache-2.0).
