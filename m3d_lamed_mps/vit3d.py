# Copyright (c) 2026 Cheng Cheng (Benjamin Cheng). MIT License, see LICENSE.
# Adapted from MONAI 1.3.0 (Copyright (c) MONAI Consortium, Apache-2.0): the ViT computation is
# re-implemented in pure PyTorch (modified). See THIRD_PARTY_NOTICES.md.
"""Pure-PyTorch 3D ViT used by M3D-LaMed (vision tower and SegVol image encoder).

The upstream checkpoint was trained with MONAI 1.3.0's ``ViT`` / ``PatchEmbeddingBlock`` /
``TransformerBlock`` / ``SABlock`` / ``MLPBlock``. Those classes changed constructor arguments
(``pos_embed`` -> ``proj_type`` was removed in MONAI 1.4) and internals across releases, so
this module re-implements exactly the MONAI 1.3.0 computation with *identical parameter
names*. That keeps the checkpoint keys stable and removes MONAI as a runtime dependency.

Only the configuration actually used by M3D-LaMed is supported:
3D input, ``proj_type="perceptron"``, learnable position embeddings, GELU MLP,
no qkv bias, optional ``cls_token`` (no classification head).

Differences from MONAI 1.3.0 (numerically equivalent up to floating-point rounding):
* the einops ``Rearrange`` modules are replaced with ``view``/``permute``;
* attention uses ``torch.nn.functional.scaled_dot_product_attention`` (a fused Metal kernel on
  MPS) instead of two ``einsum`` calls + softmax. ``use_sdpa=False`` restores the explicit path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_3tuple(v: Sequence[int] | int) -> tuple[int, int, int]:
    if isinstance(v, int):
        return (v, v, v)
    v = tuple(int(x) for x in v)
    if len(v) != 3:
        raise ValueError(f"expected 3 spatial dims, got {v}")
    return v  # type: ignore[return-value]


class PerceptronPatchify(nn.Module):
    """MONAI ``Rearrange("b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)")``."""

    def __init__(self, patch_size: Sequence[int]):
        super().__init__()
        self.patch_size = _as_3tuple(patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, s1, s2, s3 = x.shape
        p1, p2, p3 = self.patch_size
        g1, g2, g3 = s1 // p1, s2 // p2, s3 // p3
        x = x.reshape(b, c, g1, p1, g2, p2, g3, p3)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)  # b g1 g2 g3 p1 p2 p3 c
        return x.reshape(b, g1 * g2 * g3, p1 * p2 * p3 * c)


class PatchEmbeddingBlock(nn.Module):
    """MONAI 1.3.0 ``PatchEmbeddingBlock(proj_type="perceptron", pos_embed_type="learnable")``."""

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int,
        num_heads: int,
        proj_type: str = "perceptron",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        if spatial_dims != 3:
            raise NotImplementedError("only 3D inputs are supported")
        if proj_type != "perceptron":
            raise NotImplementedError("M3D-LaMed uses proj_type='perceptron' only")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        img_size = _as_3tuple(img_size)
        patch_size = _as_3tuple(patch_size)
        for m, p in zip(img_size, patch_size):
            if m % p != 0:
                raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = math.prod(m // p for m, p in zip(img_size, patch_size))
        self.patch_dim = in_channels * math.prod(patch_size)
        # index 0 has no parameters, so the Linear is "patch_embeddings.1.*" exactly as in MONAI
        self.patch_embeddings = nn.Sequential(
            PerceptronPatchify(patch_size), nn.Linear(self.patch_dim, hidden_size)
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[2:]) != self.img_size:
            raise ValueError(f"expected input spatial size {self.img_size}, got {tuple(x.shape[2:])}")
        x = self.patch_embeddings(x)
        return self.dropout(x + self.position_embeddings)


class SABlock(nn.Module):
    """MONAI 1.3.0 ``SABlock`` (multi-head self-attention)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.use_sdpa = use_sdpa

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        # MONAI: Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.use_sdpa:
            drop = self.drop_weights.p if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, scale=self.scale)
        else:
            att = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
            out = self.drop_weights(att) @ v
        # MONAI: Rearrange("b h l d -> b l (h d)")
        out = out.transpose(1, 2).reshape(b, n, self.num_heads * self.head_dim)
        return self.drop_output(self.out_proj(out))


class MLPBlock(nn.Module):
    """MONAI 1.3.0 ``MLPBlock`` (act="GELU", dropout_mode="vit")."""

    def __init__(self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        mlp_dim = mlp_dim or hidden_size
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.fn = nn.GELU()  # exact (erf) GELU, as MONAI's get_act_layer("GELU")
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.linear2(self.drop1(self.fn(self.linear1(x)))))


class TransformerBlock(nn.Module):
    """MONAI 1.3.0 ``TransformerBlock`` (pre-norm)."""

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock(hidden_size, num_heads, dropout_rate, qkv_bias, use_sdpa=use_sdpa)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """MONAI 1.3.0 ``ViT`` without the classification head.

    With ``classification=True`` a ``cls_token`` is prepended (M3D's ``myViT``); with
    ``classification=False`` it is not (SegVol image encoder). Returns
    ``(normed_last_hidden_state, [hidden_state_after_each_block])`` like MONAI.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        proj_type: str = "perceptron",
        classification: bool = False,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1")
        self.hidden_size = hidden_size
        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=proj_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, use_sdpa=use_sdpa)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def set_use_sdpa(self, enabled: bool) -> None:
        for blk in self.blocks:
            blk.attn.use_sdpa = enabled

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.patch_embedding(x)
        if self.classification:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        return self.norm(x), hidden_states_out
