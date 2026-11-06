# Copyright (c) 2026 Cheng Cheng (Benjamin Cheng). MIT License, see LICENSE.
# Adapted from the M3D-LaMed remote code (GoodBaiBai88/M3D-LaMed-Phi-3-4B, Apache-2.0), SegVol
# (Copyright (c) 2023 BAAI-DCAI, MIT) and Segment Anything (Copyright (c) Meta Platforms, Inc. and
# affiliates, Apache-2.0); modified for Apple MPS. See THIRD_PARTY_NOTICES.md.
"""SegVol (SAM-style 3D) segmentation module of M3D-LaMed, ported for Apple MPS.

Derived from ``modeling_m3d_lamed.py`` of ``GoodBaiBai88/M3D-LaMed-Phi-3-4B`` (revision
329bed510c2cff0ad3fe6f2a1a7075f45e2706e9), Apache-2.0. Module/parameter names are unchanged
so the original checkpoint loads 1:1.

MPS-relevant changes (see PORTING_NOTES.md):
* ``nn.ConvTranspose3d`` in the mask decoder is replaced by :class:`PatchwiseConvTranspose3d`,
  an exact matmul + reshape formulation for ``kernel_size == stride`` (PyTorch 2.15 MPS raises
  "ConvTranspose 3D with BF16 or FP16 types is not supported on MPS"). Parameters are identical.
* Cross-attention uses ``F.scaled_dot_product_attention`` (same math; ``use_sdpa=False`` restores
  the explicit softmax path).
* The image encoder is the vendored MONAI-1.3.0-equivalent :class:`~m3d_lamed_mps.vit3d.ViT`.
* Unused upstream code (2D SAM ``ImageEncoderViT``/window attention, the ``Sam`` wrapper, training
  losses, debug ``print`` calls) is removed.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .vit3d import ViT


class PatchwiseConvTranspose3d(nn.ConvTranspose3d):
    """``nn.ConvTranspose3d`` for the non-overlapping case ``kernel_size == stride``.

    Each input voxel writes an independent ``k1 x k2 x k3`` output block, so the transposed
    convolution is exactly ``x @ W.reshape(C_in, C_out * k1*k2*k3)`` followed by a reshape.
    This runs on MPS in fp32/fp16/bf16 (native ConvTranspose3d on MPS is fp32-only).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            tuple(self.kernel_size) != tuple(self.stride)
            or any(self.padding)
            or any(self.output_padding)
            or any(d != 1 for d in self.dilation)
            or self.groups != 1
            or self.padding_mode != "zeros"
        ):
            raise ValueError("PatchwiseConvTranspose3d requires kernel_size == stride, no padding/dilation/groups")

    def forward(self, x: Tensor, output_size=None) -> Tensor:  # noqa: D401
        if output_size is not None:
            raise ValueError("output_size is not supported")
        b, c_in, d, h, w = x.shape
        k1, k2, k3 = self.kernel_size
        c_out = self.out_channels
        weight = self.weight.reshape(c_in, c_out * k1 * k2 * k3)
        y = x.permute(0, 2, 3, 4, 1).reshape(b * d * h * w, c_in) @ weight
        y = y.view(b, d, h, w, c_out, k1, k2, k3).permute(0, 4, 1, 5, 2, 6, 3, 7)
        y = y.reshape(b, c_out, d * k1, h * k2, w * k3)
        if self.bias is not None:
            y = y + self.bias.view(1, c_out, 1, 1, 1)
        return y


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MLPBlock(nn.Module):
    """SAM MLP block (``lin1``/``lin2``); not the MONAI one."""

    def __init__(self, embedding_dim: int, mlp_dim: int, act: type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: Tensor) -> Tensor:
        return self.lin2(self.act(self.lin1(x)))


class Attention(nn.Module):
    """SAM attention with optional down-projection of q/k/v."""

    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1, use_sdpa: bool = True) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        if self.internal_dim % num_heads != 0:
            raise ValueError("num_heads must divide embedding_dim")
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.use_sdpa = use_sdpa

    def _separate_heads(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        return x.reshape(b, n, self.num_heads, c // self.num_heads).transpose(1, 2)

    @staticmethod
    def _recombine_heads(x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        return x.transpose(1, 2).reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self._separate_heads(self.q_proj(q))
        k = self._separate_heads(self.k_proj(k))
        v = self._separate_heads(self.v_proj(v))
        if self.use_sdpa:
            out = F.scaled_dot_product_attention(q, k, v)  # scale = 1/sqrt(c_per_head), as upstream
        else:
            attn = (q @ k.permute(0, 1, 3, 2)) / math.sqrt(q.shape[-1])
            out = torch.softmax(attn, dim=-1) @ v
        return self.out_proj(self._recombine_heads(out))


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor) -> tuple[Tensor, Tensor]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(q=q, k=q, v=queries)
        queries = self.norm1(queries)

        q = queries + query_pe
        k = keys + key_pe
        queries = self.norm2(queries + self.cross_attn_token_to_image(q=q, k=k, v=keys))

        queries = self.norm3(queries + self.mlp(queries))

        q = queries + query_pe
        k = keys + key_pe
        keys = self.norm4(keys + self.cross_attn_image_to_token(q=k, k=q, v=queries))
        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList(
            TwoWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                activation=activation,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=(i == 0),
            )
            for i in range(depth)
        )
        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding: Tensor, image_pe: Tensor, point_embedding: Tensor) -> tuple[Tensor, Tensor]:
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        queries, keys = point_embedding, image_embedding
        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys, query_pe=point_embedding, key_pe=image_pe)
        q = queries + point_embedding
        k = keys + image_pe
        queries = self.norm_final_attn(queries + self.final_attn_token_to_image(q=q, k=k, v=keys))
        return queries, keys


class PositionEmbeddingRandom(nn.Module):
    """Positional encoding using random spatial frequencies (persistent buffer, loaded from ckpt)."""

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer("positional_encoding_gaussian_matrix", scale * torch.randn((3, num_pos_feats)))

    def _pe_encoding(self, coords: Tensor) -> Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: tuple[int, int, int]) -> Tensor:
        h, w, d = size
        device = self.positional_encoding_gaussian_matrix.device
        dtype = self.positional_encoding_gaussian_matrix.dtype
        grid = torch.ones((h, w, d), device=device, dtype=dtype)
        y_embed = (grid.cumsum(dim=0) - 0.5) / h
        x_embed = (grid.cumsum(dim=1) - 0.5) / w
        z_embed = (grid.cumsum(dim=2) - 0.5) / d
        pe = self._pe_encoding(torch.stack([x_embed, y_embed, z_embed], dim=-1))
        return pe.permute(3, 0, 1, 2)  # C x H x W x D

    def forward_with_coords(self, coords_input: Tensor, image_size: tuple[int, int, int]) -> Tensor:
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        coords[:, :, 2] = coords[:, :, 2] / image_size[2]
        return self._pe_encoding(coords.to(self.positional_encoding_gaussian_matrix.dtype))


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: tuple[int, int, int],
        input_image_size: tuple[int, int, int],
        mask_in_chans: int,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.num_point_embeddings = 4  # pos/neg point + 2 box corners
        self.point_embeddings = nn.ModuleList(nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings))
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.mask_input_size = tuple(4 * s for s in image_embedding_size)
        # Unused at inference (masks=None) but kept so checkpoint keys mask_downscaling.{0,1,3,4,6} load.
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points: Tensor, labels: Tensor, pad: bool) -> Tensor:
        points = points + 0.5
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 3), device=points.device, dtype=points.dtype)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device, dtype=labels.dtype)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: Tensor) -> Tensor:
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 3)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def forward(
        self,
        points: Optional[tuple[Tensor, Tensor]],
        boxes: Optional[Tensor],
        masks: Optional[Tensor],
        text_embedding: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        if points is not None:
            bs = points[0].shape[0]
        elif boxes is not None:
            bs = boxes.shape[0]
        elif masks is not None:
            bs = masks.shape[0]
        elif text_embedding is not None:
            bs = text_embedding.shape[0]
        else:
            bs = 1
        weight = self.point_embeddings[0].weight
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=weight.device, dtype=weight.dtype)
        if points is not None:
            coords, labels = points
            sparse_embeddings = torch.cat([sparse_embeddings, self._embed_points(coords, labels, pad=(boxes is None))], dim=1)
        if boxes is not None:
            sparse_embeddings = torch.cat([sparse_embeddings, self._embed_boxes(boxes)], dim=1)
        if text_embedding is not None:
            sparse_embeddings = torch.cat([sparse_embeddings, text_embedding.unsqueeze(dim=1)], dim=1)
        if masks is not None:
            raise NotImplementedError("mask prompts are not used by M3D-LaMed inference")
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1, 1).expand(
            bs, -1, *(int(s) for s in self.image_embedding_size)
        )
        return sparse_embeddings, dense_embeddings


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return torch.sigmoid(x) if self.sigmoid_output else x


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        image_encoder_type: str,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        image_size,
        patch_size,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        image_size, patch_size = np.asarray(image_size), np.asarray(patch_size)
        if image_encoder_type == "swin_vit":
            self.feat_shape = image_size / patch_size
        else:  # 'vit' (used by M3D-LaMed)
            self.feat_shape = image_size / patch_size * 2
        self.output_upscaling = nn.Sequential(
            PatchwiseConvTranspose3d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            nn.LayerNorm((transformer_dim // 4, *(int(s) for s in self.feat_shape))),
            activation(),
            PatchwiseConvTranspose3d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for _ in range(self.num_mask_tokens)
        )
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)
        self.txt_align_upscaled_embedding = nn.Linear(768, 96)

    def forward(
        self,
        image_embeddings: Tensor,
        text_embedding: Optional[Tensor],
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
        multimask_output: bool,
    ) -> tuple[Tensor, Tensor]:
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            text_embedding=text_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        mask_slice = slice(1, None) if multimask_output else slice(0, 1)
        return masks[:, mask_slice], iou_pred[:, mask_slice]

    def predict_masks(
        self,
        image_embeddings: Tensor,
        text_embedding: Optional[Tensor],
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
    ) -> tuple[Tensor, Tensor]:
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w, d = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).reshape(b, c, h, w, d)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in = torch.stack(
            [self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) for i in range(self.num_mask_tokens)], dim=1
        )
        b, c, h, w, d = upscaled_embedding.shape
        upscaled_flat = upscaled_embedding.reshape(b, c, h * w * d)
        masks = (hyper_in @ upscaled_flat).view(b, -1, h, w, d)

        if text_embedding is not None:
            text_embedding_down = self.txt_align_upscaled_embedding(text_embedding).unsqueeze(dim=1)
            sim = (text_embedding_down @ upscaled_flat).view(b, -1, h, w, d)
            masks = masks + sim.repeat(1, masks.shape[1], 1, 1, 1)
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred


class SegVol(nn.Module):
    def __init__(self, image_encoder: nn.Module, mask_decoder: MaskDecoder, prompt_encoder: PromptEncoder, roi_size, patch_size):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.feat_shape = np.array(roi_size) / np.array(patch_size)

    def forward(self, image: Tensor, text_emb: Optional[Tensor] = None, text=None, boxes=None, points=None) -> Tensor:
        bs = image.shape[0]
        img_shape = (image.shape[2], image.shape[3], image.shape[4])
        image_embedding, _ = self.image_encoder(image)
        image_embedding = image_embedding.transpose(1, 2).reshape(bs, -1, *(int(s) for s in self.feat_shape))
        return self.forward_decoder(image_embedding, img_shape, text_emb=text_emb, boxes=boxes, points=points)

    def forward_decoder(self, image_embedding: Tensor, img_shape, text_emb=None, text=None, boxes=None, points=None) -> Tensor:
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=None, masks=None, text_embedding=text_emb
        )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            text_embedding=text_emb,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return F.interpolate(low_res_masks, size=img_shape, mode="trilinear", align_corners=False)


def build_segmentation_module(config) -> SegVol:
    """Equivalent of upstream ``build_segmentation_module`` / ``_build_sam(image_encoder_type='vit')``."""
    name = getattr(config, "segmentation_module")
    if "segvol" not in str(name).lower():
        raise ValueError(f"Unknown segmentation module: {name}")
    embed_dim = 768
    image_size, patch_size = list(config.image_size), list(config.patch_size)
    image_encoder = ViT(
        in_channels=1,
        img_size=image_size,
        patch_size=patch_size,
        hidden_size=embed_dim,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        proj_type="perceptron",
        classification=False,
        dropout_rate=0.0,
    )
    image_embedding_size = [int(x) for x in (np.array(image_size) / np.array(patch_size))]
    prompt_encoder = PromptEncoder(
        embed_dim=embed_dim,
        image_embedding_size=tuple(image_embedding_size),
        input_image_size=tuple(image_size),
        mask_in_chans=16,
    )
    mask_decoder = MaskDecoder(
        image_encoder_type="vit",
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(depth=2, embedding_dim=embed_dim, mlp_dim=2048, num_heads=8),
        transformer_dim=embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        image_size=np.array(image_size),
        patch_size=np.array(patch_size),
    )
    return SegVol(
        image_encoder=image_encoder,
        mask_decoder=mask_decoder,
        prompt_encoder=prompt_encoder,
        roi_size=image_size,
        patch_size=patch_size,
    )
