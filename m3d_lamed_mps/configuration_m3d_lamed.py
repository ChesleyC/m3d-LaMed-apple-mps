# Copyright (c) 2026 Cheng Cheng (Benjamin Cheng). MIT License, see LICENSE.
# Adapted from the M3D-LaMed remote code (GoodBaiBai88/M3D-LaMed-Phi-3-4B, Apache-2.0); modified.
# See THIRD_PARTY_NOTICES.md.
"""Configuration for M3D-LaMed-Phi-3-4B.

Upstream (``configuration_m3d_lamed.py``) is a bare ``class LamedPhi3Config(Phi3Config)`` that relied
on Transformers 4.x storing unknown config keys as attributes. Transformers 5 configs are strict
dataclasses, so the multimodal fields are declared explicitly here (defaults = released checkpoint).
"""

from __future__ import annotations

from huggingface_hub.dataclasses import strict
from transformers import Phi3Config


@strict
class LamedPhi3Config(Phi3Config):
    model_type = "lamed_phi3"

    # 3D vision input / tower
    image_channel: int = 1
    image_size: list[int] | tuple[int, ...] = (32, 256, 256)
    patch_size: list[int] | tuple[int, ...] = (4, 16, 16)
    vision_tower: str | None = "vit3d"
    vision_select_layer: int = -1
    vision_select_feature: str = "patch"
    # multimodal projector
    mm_projector_type: str = "spp"
    mm_hidden_size: int = 768
    proj_layer_type: str = "mlp"
    proj_layer_num: int = 2
    proj_pooling_type: str = "spatial"
    proj_pooling_size: int = 2
    # segmentation
    segmentation_module: str | None = "segvol"
    seg_token_id: int = 32014

    @property
    def num_image_tokens(self) -> int:
        """Number of ``<im_patch>`` placeholders = projector output length (256 for the release)."""
        if self.mm_projector_type != "spp":
            raise NotImplementedError(self.mm_projector_type)
        n = 1
        for img, pch in zip(self.image_size, self.patch_size):
            n *= (img // pch) // (self.proj_pooling_size if self.proj_pooling_type == "spatial" else 1)
        if self.proj_pooling_type == "sequence":
            n //= self.proj_pooling_size**3
        return n
