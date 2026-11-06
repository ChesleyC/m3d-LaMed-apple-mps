# Third-party notices

This project is licensed under the MIT License (see [LICENSE](LICENSE)). It includes files from, and
code adapted from, the projects below. Their licences continue to apply to those parts. Adapted source
files carry a header naming what they are derived from, and the changes are listed in
[PORTING_NOTES.md](PORTING_NOTES.md#modifications).

The model weights are **not** part of this repository. `scripts/download_weights.sh` downloads them
from Hugging Face (`GoodBaiBai88/M3D-LaMed-Phi-3-4B`, Apache-2.0).

| Component | Where it is used here | Licence | Copyright |
|---|---|---|---|
| [M3D](https://github.com/BAAI-DCAI/M3D) | model code adapted in `m3d_lamed_mps/modeling_m3d_lamed.py` and `segvol.py`; prompt templates copied verbatim in `prompts.py`; preprocessing reproduced in `preprocessing.py` | MIT | (c) 2024 BAAI-DCAI |
| [M3D-LaMed-Phi-3-4B model files](https://huggingface.co/GoodBaiBai88/M3D-LaMed-Phi-3-4B) | `upstream/hf_M3D-LaMed-Phi-3-4B/`: config, tokenizer and original remote-code files, unmodified; `safetensors_headers/` extracted from the released shards. The remote code is adapted in `modeling_m3d_lamed.py`, `configuration_m3d_lamed.py` and `segvol.py` | Apache-2.0 | the M3D-LaMed authors |
| [SegVol](https://github.com/BAAI-DCAI/SegVol) | segmentation module adapted (via the M3D-LaMed code) in `segvol.py` | MIT | (c) 2023 BAAI-DCAI |
| [Segment Anything](https://github.com/facebookresearch/segment-anything) | prompt encoder, two-way transformer and mask decoder design adapted (via SegVol) in `segvol.py` | Apache-2.0 | (c) Meta Platforms, Inc. and affiliates |
| [MONAI](https://github.com/Project-MONAI/MONAI) 1.3.0 | `ViT` / `PatchEmbeddingBlock` / `TransformerBlock` / `SABlock` / `MLPBlock` computation re-implemented in `vit3d.py`; `CropForeground` / `Resize` behaviour reproduced in `preprocessing.py` | Apache-2.0 | (c) MONAI Consortium |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) 4.42.4 | slow `LlamaTokenizer` / `PreTrainedTokenizer` algorithm ported in `tokenizer.py` | Apache-2.0 | (c) The HuggingFace Inc. team |
| [Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | tokenizer files inherited by M3D-LaMed (`tokenizer.model` etc. in `upstream/hf_M3D-LaMed-Phi-3-4B/`) | MIT | (c) Microsoft Corporation |

None of these projects ships an Apache `NOTICE` file. The full Apache License 2.0 text is in
[third_party_licenses/Apache-2.0.txt](third_party_licenses/Apache-2.0.txt). The MIT notices are reproduced below as published.

## BAAI-DCAI/M3D

```
MIT License

Copyright (c) 2024 BAAI-DCAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## BAAI-DCAI/SegVol

```
MIT License

Copyright (c) 2023 BAAI-DCAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## microsoft/Phi-3-mini-4k-instruct

```
Microsoft.
Copyright (c) Microsoft Corporation.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
