"""The vendored ViT must be a drop-in replacement for MONAI's (same keys, same math).

MONAI (and einops) are test-only dependencies; the runtime package does not import them.
"""

import pytest
import torch

from m3d_lamed_mps.vit3d import PerceptronPatchify, ViT
from tests.conftest import rel_err

monai = pytest.importorskip("monai")
einops = pytest.importorskip("einops")

IMG, PATCH = (32, 256, 256), (4, 16, 16)


def _monai_vit(classification: bool):
    from monai.networks.nets import ViT as MonaiViT

    m = MonaiViT(in_channels=1, img_size=IMG, patch_size=PATCH, hidden_size=768, mlp_dim=3072, num_layers=12,
                 num_heads=12, proj_type="perceptron", classification=classification, dropout_rate=0.0)
    if classification:
        del m.classification_head  # M3D's myViT has no head
    return m.eval()


def test_patchify_matches_einops_rearrange():
    from einops.layers.torch import Rearrange

    x = torch.randn(2, 1, *IMG)
    ref = Rearrange("b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)", p1=4, p2=16, p3=16)(x)
    assert torch.equal(PerceptronPatchify(PATCH)(x), ref)


@pytest.mark.parametrize("classification", [True, False], ids=["vision_tower(cls)", "segvol_encoder"])
def test_vit_matches_monai(classification):
    torch.manual_seed(0)
    ref = _monai_vit(classification)
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.02)
    ours = ViT(in_channels=1, img_size=IMG, patch_size=PATCH, classification=classification).eval()
    ref_sd = ref.state_dict()
    # MONAI >= 1.4 always creates an (unused without cross-attention) ``norm_cross_attn`` LayerNorm in
    # every TransformerBlock. The M3D checkpoint (trained with MONAI 1.3.0) has no such tensors - one
    # of the version-drift problems the vendored ViT avoids. Everything else must match 1:1.
    extra = set(ref_sd) - set(ours.state_dict())
    assert all(".norm_cross_attn." in k for k in extra), extra
    assert set(ours.state_dict()) <= set(ref_sd)
    ours.load_state_dict({k: v for k, v in ref_sd.items() if k not in extra}, strict=True)

    x = torch.rand(1, 1, *IMG)
    with torch.no_grad():
        y_ref, hs_ref = ref(x)
        for use_sdpa in (False, True):
            ours.set_use_sdpa(use_sdpa)
            y, hs = ours(x)
            assert y.shape == y_ref.shape == ((1, 2049, 768) if classification else (1, 2048, 768))
            assert rel_err(y, y_ref) < 1e-5, (use_sdpa, rel_err(y, y_ref))
            assert len(hs) == len(hs_ref) == 12
            assert rel_err(hs[5], hs_ref[5]) < 1e-5
