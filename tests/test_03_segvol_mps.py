"""SegVol segmentation module on MPS (synthetic tensors), incl. the ConvTranspose3d replacement."""

import copy

import pytest
import torch
import torch.nn as nn

from m3d_lamed_mps.mps_utils import DeviceAudit
from m3d_lamed_mps.segvol import PatchwiseConvTranspose3d, build_segmentation_module
from tests.conftest import cos_sim, rel_err


@pytest.mark.parametrize("cin,cout,shape", [(768, 192, (8, 16, 16)), (192, 96, (16, 32, 32)), (5, 3, (2, 3, 4))])
def test_patchwise_convtranspose_matches_native_cpu(cin, cout, shape):
    torch.manual_seed(0)
    native = nn.ConvTranspose3d(cin, cout, kernel_size=2, stride=2)
    ours = PatchwiseConvTranspose3d(cin, cout, kernel_size=2, stride=2)
    ours.load_state_dict(native.state_dict())
    x = torch.randn(2, cin, *shape)
    with torch.no_grad():
        assert rel_err(ours(x), native(x)) < 1e-6


def test_patchwise_convtranspose_on_mps(mps):
    torch.manual_seed(0)
    native = nn.ConvTranspose3d(768, 192, 2, 2)
    ours = PatchwiseConvTranspose3d(768, 192, 2, 2)
    ours.load_state_dict(native.state_dict())
    x = torch.randn(1, 768, 8, 16, 16)
    with torch.no_grad():
        ref = native(x)
        assert rel_err(ours.to(mps)(x.to(mps)), ref) < 1e-5  # fp32 on MPS
        assert rel_err(native.to(mps)(x.to(mps)), ref) < 1e-5  # native fp32 MPS kernel exists
        for dt in (torch.bfloat16, torch.float16):
            y = copy.deepcopy(ours).to(mps, dt)(x.to(mps, dt))
            assert y.dtype == dt and y.device.type == "mps"
            assert cos_sim(y, ref) > 0.999


def test_native_low_precision_convtranspose3d_still_unsupported_on_mps(mps):
    """Documents *why* the replacement exists. If a future PyTorch adds the kernel this test fails,
    which is the signal that the workaround could be dropped."""
    m = nn.ConvTranspose3d(8, 4, 2, 2).to(mps, torch.bfloat16)
    with pytest.raises(RuntimeError, match="ConvTranspose 3D"):
        m(torch.randn(1, 8, 2, 2, 2, device=mps, dtype=torch.bfloat16))


def test_patchwise_rejects_overlapping_kernels():
    with pytest.raises(ValueError):
        PatchwiseConvTranspose3d(4, 4, kernel_size=3, stride=2)


@pytest.fixture(scope="module")
def segvol_cpu(real_config):
    torch.manual_seed(0)
    m = build_segmentation_module(real_config).eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "norm" in name and name.endswith("weight"):
                p.fill_(1.0)
            else:
                p.normal_(0, 0.02)
    return m


def test_segvol_on_mps_matches_cpu(mps, segvol_cpu):
    torch.manual_seed(2)
    image = torch.rand(1, 1, 32, 256, 256)
    text_emb = torch.randn(1, 768)
    with torch.no_grad():
        ref = segvol_cpu(image, text_emb=text_emb)
    assert ref.shape == (1, 1, 32, 256, 256)
    for dt, min_cos in ((torch.float32, 0.99999), (torch.bfloat16, 0.99)):
        m = copy.deepcopy(segvol_cpu).to(mps, dt)
        audit = DeviceAudit()
        with torch.no_grad(), audit:
            out = m(image.to(mps, dt), text_emb=text_emb.to(mps, dt))
        torch.mps.synchronize()
        assert out.shape == (1, 1, 32, 256, 256) and out.device.type == "mps" and out.dtype == dt
        assert torch.isfinite(out).all()
        assert audit.report.summary()["ok"], audit.report.summary()
        assert cos_sim(out, ref) > min_cos, (dt, cos_sim(out, ref))


def test_seg_attention_sdpa_equals_explicit(segvol_cpu):
    attn = segvol_cpu.mask_decoder.transformer.layers[0].cross_attn_token_to_image
    q, k = torch.randn(2, 6, 768), torch.randn(2, 2048, 768)
    with torch.no_grad():
        attn.use_sdpa = True
        a = attn(q, k, k)
        attn.use_sdpa = False
        b = attn(q, k, k)
        attn.use_sdpa = True
    assert rel_err(a, b) < 1e-5
