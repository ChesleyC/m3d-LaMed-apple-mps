# Copyright (c) 2026 Cheng Cheng (Benjamin Cheng). MIT License, see LICENSE.
# Adapted from the M3D-LaMed remote code (GoodBaiBai88/M3D-LaMed-Phi-3-4B, Apache-2.0) and
# BAAI-DCAI/M3D (Copyright (c) 2024 BAAI-DCAI, MIT); modified for Apple MPS and Transformers 5.
# See THIRD_PARTY_NOTICES.md.
"""M3D-LaMed-Phi-3-4B (inference) ported to Apple Silicon / PyTorch MPS and Transformers 5.

Derived from ``modeling_m3d_lamed.py`` of ``GoodBaiBai88/M3D-LaMed-Phi-3-4B`` (revision
329bed510c2cff0ad3fe6f2a1a7075f45e2706e9, Apache-2.0) and ``LaMed/src/model`` of BAAI-DCAI/M3D (MIT).
All module/parameter names are unchanged, so the released safetensors load 1:1 (617 tensors).

Inference semantics preserved from upstream:
* prompt = ``<s>`` + ``"<im_patch>" * 256`` + question; embeddings at positions ``1..256`` are
  replaced by the projected 3D-ViT features (``prepare_inputs_for_multimodal``);
* ``generate(images, input_ids, seg_enable=False, **gen_kwargs)`` returns generated token ids
  (``seg_enable=True`` additionally returns SegVol logits ``B x 1 x 32 x 256 x 256``);
* the [SEG] hidden-state selection (``output_ids[:, 1:] == seg_token_id``) is identical.

Changes are listed in PORTING_NOTES.md. The most important: no hard-coded CUDA device calls,
no MONAI/einops runtime dependency, Transformers-5 compatible forward/generate, an explicit check
that the ``<im_patch>`` placeholders are where the image features are written, and an
MPS-native ConvTranspose3d in SegVol.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Phi3ForCausalLM, Phi3Model
from transformers.models.phi3.modeling_phi3 import Phi3PreTrainedModel

from .configuration_m3d_lamed import LamedPhi3Config
from .segvol import build_segmentation_module
from .vit3d import ViT

IMAGE_TOKEN = "<im_patch>"
IMAGE_TOKEN_ID = 32011  # added_tokens.json of the released checkpoint
SEG_TOKEN = "[SEG]"


class SpatialPoolingProjector(nn.Module):
    """``mm_projector_type='spp'``: 2x2x2 average pooling of the ViT patch grid + MLP."""

    def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type, layer_num, pooling_type="spatial", pooling_size=2):
        super().__init__()
        self.in_dim = in_dim
        self.pooling_size = pooling_size
        self.pooling_type = pooling_type
        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]
        self.num_patches_post = [num // pooling_size for num in self.num_patches_pre]

        depth = int(layer_num)
        if layer_type == "linear":
            modules = [nn.Linear(in_dim, out_dim)] + [nn.Linear(out_dim, out_dim) for _ in range(1, depth)]
        elif layer_type == "mlp":
            modules = [nn.Linear(in_dim, out_dim)]
            for _ in range(1, depth):
                modules += [nn.GELU(), nn.Linear(out_dim, out_dim)]
        else:
            raise ValueError(f"Unknown projector layer type: {layer_type}")
        self.projector = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        if self.pooling_type == "spatial":
            # einops "b (p1 p2 p3) d -> b d p1 p2 p3", avg_pool3d, "b d p1 p2 p3 -> b (p1 p2 p3) d"
            x = x.transpose(1, 2).reshape(b, d, *self.num_patches_pre)
            x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)
            x = x.flatten(2).transpose(1, 2)
        elif self.pooling_type == "sequence":
            k = self.pooling_size**3
            x = F.avg_pool1d(x.transpose(1, 2), kernel_size=k, stride=k).transpose(1, 2)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
        return self.projector(x)

    @property
    def proj_out_num(self) -> int:
        num = 1
        for n in self.num_patches_post:
            num *= n
        return num


class IdentityMap(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


def build_mm_projector(config: LamedPhi3Config) -> nn.Module:
    projector_type = config.mm_projector_type
    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    if projector_type == "spp":
        return SpatialPoolingProjector(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_dim=config.mm_hidden_size,
            out_dim=config.hidden_size,
            layer_type=config.proj_layer_type,
            layer_num=config.proj_layer_num,
            pooling_type=config.proj_pooling_type,
            pooling_size=config.proj_pooling_size,
        )
    if projector_type == "identity":
        return IdentityMap()
    raise ValueError(f"Unknown projector type: {projector_type}")


class ViT3DTower(nn.Module):
    """3D ViT vision tower (M3D-CLIP initialised). Output: patch features ``B x 2048 x 768``."""

    def __init__(self, config: LamedPhi3Config):
        super().__init__()
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature
        self.vision_tower = ViT(
            in_channels=config.image_channel,
            img_size=config.image_size,
            patch_size=config.patch_size,
            proj_type="perceptron",
            spatial_dims=len(config.patch_size),
            classification=True,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        last_feature, hidden_states = self.vision_tower(images)
        if self.select_layer == -1:
            image_features = last_feature
        elif self.select_layer < -1:
            # upstream indexed with ``self.select_feature`` (a str) here, which would raise
            image_features = hidden_states[self.select_layer]
        else:
            raise ValueError(f"Unexpected select layer: {self.select_layer}")
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature != "cls_patch":
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def hidden_size(self) -> int:
        return self.vision_tower.hidden_size


def build_vision_tower(config: LamedPhi3Config) -> ViT3DTower:
    if "vit3d" in str(config.vision_tower).lower():
        return ViT3DTower(config)
    raise ValueError(f"Unknown vision tower: {config.vision_tower}")


class LamedPhi3Model(Phi3Model):
    """Phi-3 decoder + ``vision_tower``, ``mm_projector``, ``seg_module``, ``seg_projector``."""

    config_class = LamedPhi3Config

    def __init__(self, config: LamedPhi3Config):
        super().__init__(config)
        self.seg_enable = False
        if getattr(config, "vision_tower", None):
            self.vision_tower = build_vision_tower(config)
            self.mm_projector = build_mm_projector(config)
        if getattr(config, "segmentation_module", None):
            self.seg_enable = True
            self.seg_module = build_segmentation_module(config)
            self.seg_projector = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(config.hidden_size, config.mm_hidden_size),
                nn.Dropout(0.1),
            )

    def get_vision_tower(self) -> Optional[ViT3DTower]:
        return getattr(self, "vision_tower", None)


class LamedPhi3ForCausalLM(Phi3ForCausalLM):
    config_class = LamedPhi3Config

    def __init__(self, config: LamedPhi3Config):
        # Skip Phi3ForCausalLM.__init__ (it would build and then discard a plain Phi3Model).
        Phi3PreTrainedModel.__init__(self, config)
        self.model = LamedPhi3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.image_token_id = IMAGE_TOKEN_ID
        self.post_init()

    # ------------------------------------------------------------------ helpers
    def get_model(self) -> LamedPhi3Model:
        return self.model

    def get_vision_tower(self) -> Optional[ViT3DTower]:
        return self.model.get_vision_tower()

    @property
    def num_image_tokens(self) -> int:
        return self.config.num_image_tokens

    def set_vision_sdpa(self, enabled: bool) -> None:
        """Toggle fused SDPA vs explicit softmax attention in the 3D ViTs and SegVol decoder."""
        for m in self.modules():
            if hasattr(m, "use_sdpa"):
                m.use_sdpa = enabled

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """``B x 1 x 32 x 256 x 256`` in [0, 1] -> image token embeddings ``B x 256 x hidden``."""
        tower = self.get_vision_tower()
        images = images.to(device=tower.device, dtype=tower.dtype)
        return self.model.mm_projector(tower(images))

    def embed_multimodal_inputs(self, input_ids: torch.LongTensor, images: torch.Tensor, check_image_tokens: bool = True) -> torch.Tensor:
        """Upstream ``prepare_inputs_for_multimodal``: write image features into positions 1..N."""
        image_features = self.encode_images(images)
        n = image_features.shape[1]
        if check_image_tokens:
            if input_ids.shape[1] < n + 1:
                raise ValueError(f"prompt has {input_ids.shape[1]} tokens; need BOS + {n} {IMAGE_TOKEN} tokens")
            placeholder = input_ids[:, 1 : n + 1]
            if not bool((placeholder == self.image_token_id).all()):
                raise ValueError(
                    f"input_ids[:, 1:{n + 1}] must all be {IMAGE_TOKEN} (id {self.image_token_id}); "
                    f"build the prompt as tokenizer('{IMAGE_TOKEN}' * {n} + question) with BOS"
                )
        inputs_embeds = self.model.embed_tokens(input_ids)
        image_features = image_features.to(inputs_embeds.dtype)
        return torch.cat((inputs_embeds[:, :1, :], image_features, inputs_embeds[:, n + 1 :, :]), dim=1)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        images: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Any = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ):
        if inputs_embeds is None and images is not None and input_ids is not None and input_ids.shape[1] != 1:
            inputs_embeds = self.embed_multimodal_inputs(input_ids, images)
            input_ids = None
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

    # ------------------------------------------------------------------ generate
    @torch.no_grad()
    def generate(
        self,
        images: Optional[torch.Tensor] = None,
        inputs: Optional[torch.LongTensor] = None,
        seg_enable: bool = False,
        **kwargs,
    ):
        """Same call convention as upstream: ``model.generate(image, input_ids, seg_enable=..., ...)``.

        Returns generated token ids only (the prompt is passed as embeddings), or
        ``(output_ids, seg_logits)`` when ``seg_enable=True``.
        """
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if inputs is None:
            inputs = kwargs.pop("input_ids", None)
        if inputs is None:
            raise ValueError("input_ids are required")
        kwargs.pop("position_ids", None)  # upstream ignored them as well

        if images is not None:
            inputs_embeds = self.embed_multimodal_inputs(inputs, images)
        else:
            inputs_embeds = self.model.embed_tokens(inputs)

        if not seg_enable:
            return super().generate(inputs_embeds=inputs_embeds, **kwargs)

        if images is None:
            raise ValueError("seg_enable=True requires images")
        if not self.model.seg_enable:
            raise ValueError("this checkpoint has no segmentation module")
        kwargs.pop("return_dict_in_generate", None)
        kwargs.pop("output_hidden_states", None)
        outputs = super().generate(
            inputs_embeds=inputs_embeds, output_hidden_states=True, return_dict_in_generate=True, **kwargs
        )
        output_ids = outputs.sequences
        seg_token_mask = output_ids[:, 1:] == self.config.seg_token_id
        steps = outputs.hidden_states[1:]
        last_hidden_state = torch.cat([step[-1] for step in steps], dim=1) if len(steps) else None

        seg_prompts, noseg_ids = [], []
        for i in range(output_ids.shape[0]):
            n_seg = int(seg_token_mask[i].sum()) if last_hidden_state is not None else 0
            if n_seg >= 1:
                seg_tokens = last_hidden_state[i][seg_token_mask[i]]
                if n_seg > 1:
                    seg_tokens = seg_tokens.mean(dim=0, keepdim=True)
                seg_prompt = self.model.seg_projector(seg_tokens)
            else:
                noseg_ids.append(i)
                seg_prompt = torch.zeros(
                    [1, self.config.mm_hidden_size], dtype=inputs_embeds.dtype, device=inputs_embeds.device
                )
            seg_prompts.append(seg_prompt)
        seg_prompts = torch.cat(seg_prompts, dim=0)

        seg_module = self.model.seg_module
        p = next(seg_module.parameters())
        logits = seg_module(images.to(device=p.device, dtype=p.dtype), seg_prompts.to(p.dtype))
        if noseg_ids:
            logits[noseg_ids] = -torch.inf
        return output_ids, logits
