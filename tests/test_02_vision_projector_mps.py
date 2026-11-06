"""Phase-1 core: 3D vision encoder + multimodal projector on MPS with synthetic tensors."""

import pytest
import torch
import torch.nn.functional as F

from m3d_lamed_mps.modeling_m3d_lamed import SpatialPoolingProjector, ViT3DTower, build_mm_projector
from m3d_lamed_mps.mps_utils import DeviceAudit
from tests.conftest import cos_sim, rel_err


@pytest.fixture(scope="module")
def vision_cpu(real_config):
    torch.manual_seed(0)
    tower = ViT3DTower(real_config).eval()
    proj = build_mm_projector(real_config).eval()
    with torch.no_grad():
        for m in (tower, proj):
            for name, p in m.named_parameters():
                if name.endswith("weight") and "norm" in name:
                    p.fill_(1.0)
                elif name.endswith("bias"):
                    p.normal_(0, 0.01)
                else:
                    p.normal_(0, 0.02)
    return tower, proj


@pytest.fixture(scope="module")
def image():
    torch.manual_seed(1)
    return torch.rand(1, 1, 32, 256, 256)


@pytest.fixture(scope="module")
def reference(vision_cpu, image):
    tower, proj = vision_cpu
    with torch.no_grad():
        feats = tower(image)
        return feats, proj(feats)


def _to(mods, device, dtype):
    import copy

    return [copy.deepcopy(m).to(device=device, dtype=dtype) for m in mods]


def test_projector_geometry(real_config):
    proj = build_mm_projector(real_config)
    assert isinstance(proj, SpatialPoolingProjector)
    assert proj.num_patches_pre == [8, 16, 16] and proj.num_patches_post == [4, 8, 8]
    assert proj.proj_out_num == real_config.num_image_tokens == 256


def test_cpu_reference_shapes(reference):
    feats, tokens = reference
    assert feats.shape == (1, 2048, 768)
    assert tokens.shape == (1, 256, 3072)
    assert torch.isfinite(tokens).all()


def test_projector_matches_upstream_einops_path(vision_cpu, reference):
    from einops import rearrange
    from einops.layers.torch import Rearrange

    _, proj = vision_cpu
    feats, tokens = reference
    x = Rearrange("b (p1 p2 p3) d -> b d p1 p2 p3", p1=8, p2=16, p3=16)(feats)
    x = F.avg_pool3d(x, kernel_size=2, stride=2)
    x = Rearrange("b d p1 p2 p3 -> b (p1 p2 p3) d", p1=4, p2=8, p3=8)(x)
    x = rearrange(proj.projector(rearrange(x, "b n d -> (b n) d")), "(b n) d -> b n d", b=1)
    assert rel_err(tokens, x) < 1e-6


@pytest.mark.parametrize("dtype,max_rel,min_cos", [
    (torch.float32, 1e-4, 0.99999),
    (torch.bfloat16, 5e-2, 0.998),
    (torch.float16, 1e-2, 0.9999),
], ids=["fp32", "bf16", "fp16"])
def test_vision_projector_on_mps(mps, vision_cpu, image, reference, dtype, max_rel, min_cos):
    tower, proj = _to(vision_cpu, mps, dtype)
    for m in (tower, proj):
        assert all(p.device.type == "mps" and p.dtype == dtype for p in m.parameters())
    audit = DeviceAudit()
    with torch.no_grad(), audit:
        feats = tower(image.to(mps, dtype))
        tokens = proj(feats)
    torch.mps.synchronize()
    assert feats.device.type == tokens.device.type == "mps"
    assert feats.shape == (1, 2048, 768) and tokens.shape == (1, 256, 3072)
    assert torch.isfinite(tokens).all()
    s = audit.report.summary()
    assert s["ok"] and not s["cpu_compute_ops"], s
    ref_feats, ref_tokens = reference
    assert rel_err(feats, ref_feats) < max_rel, rel_err(feats, ref_feats)
    assert rel_err(tokens, ref_tokens) < max_rel, rel_err(tokens, ref_tokens)
    assert cos_sim(tokens, ref_tokens) > min_cos


def test_batch_of_two_on_mps(mps, vision_cpu):
    tower, proj = _to(vision_cpu, mps, torch.bfloat16)
    x = torch.rand(2, 1, 32, 256, 256, device=mps, dtype=torch.bfloat16)
    with torch.no_grad():
        y = proj(tower(x))
        y0 = proj(tower(x[:1]))
    assert y.shape == (2, 256, 3072)
    assert cos_sim(y[:1], y0) > 0.999


def test_sdpa_equals_explicit_attention_on_mps(mps, vision_cpu, image):
    (tower,) = _to(vision_cpu[:1], mps, torch.float32)
    x = image.to(mps)
    with torch.no_grad():
        tower.vision_tower.set_use_sdpa(True)
        a = tower(x)
        tower.vision_tower.set_use_sdpa(False)
        b = tower(x)
    assert rel_err(a, b) < 1e-4
