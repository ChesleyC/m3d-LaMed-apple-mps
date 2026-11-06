"""MRI preprocessing: orientation handling, MONAI-equivalent crop/resize, [0,1] output."""

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from m3d_lamed_mps.preprocessing import (TARGET_SHAPE, crop_foreground, normalize_intensity, preprocess_volume,
                                         resize_trilinear, save_montage)
from tests.conftest import UPSTREAM_REPO


def _phantom_lps():
    """Head-like ellipsoid in a zero background + a bright cube in the patient's
    Right-Anterior-Superior octant. Array [z, y, x] with LPS identity direction
    (x -> Left, y -> Posterior, z -> Superior)."""
    z, y, x = np.mgrid[0:60, 0:90, 0:80].astype(np.float32)
    head = (((z - 30) / 28) ** 2 + ((y - 45) / 40) ** 2 + ((x - 40) / 36) ** 2) <= 1
    vol = np.where(head, 100.0, 0.0).astype(np.float32)
    vol += np.random.default_rng(0).normal(0, 1, vol.shape).astype(np.float32) * head
    # superior (high z), anterior (low y), right (low x); ~1% of voxels so it survives the 99.5th-pct clip
    vol[44:56, 12:32, 10:30] = 400.0
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((1.0, 1.0, 2.0))
    img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return img


def _marker_center(v):
    v = v[0] if v.ndim == 4 else v
    idx = np.argwhere(v > 0.8 * v.max())
    return idx.mean(axis=0)  # (slice, row, col)


@pytest.fixture(scope="module")
def phantom_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("phantom")
    base = _phantom_lps()
    files = {}
    # the same physical image stored with different voxel orders / direction matrices
    for code in ("LPS", "RAS", "ASL", "SPR", "IRP"):
        p = d / f"phantom_{code}.nii.gz"
        sitk.WriteImage(sitk.DICOMOrient(base, code), str(p))
        files[code] = p
    return files


def test_radiological_orientation(phantom_files):
    vol, info = preprocess_volume(phantom_files["RAS"], orientation="radiological")
    assert vol.shape == (1, *TARGET_SHAPE) and vol.dtype == np.float32
    assert 0.0 <= vol.min() and vol.max() <= 1.0
    s, r, c = _marker_center(vol)
    assert s > 16, "slices run inferior -> superior: superior marker in later slices"
    assert r < 128, "rows anterior -> posterior: anterior marker near the top"
    assert c < 128, "cols patient-right -> left: right-sided marker on the image left"
    assert info.crop_box and info.orientation == "radiological"


def test_ras_orientation_matches_official_examples_layout(phantom_files):
    vol, _ = preprocess_volume(phantom_files["RAS"], orientation="ras")
    s, r, c = _marker_center(vol)
    assert s > 16 and r > 128 and c > 128


def test_storage_order_does_not_matter(phantom_files):
    ref, _ = preprocess_volume(phantom_files["LPS"])
    for code, p in phantom_files.items():
        out, _ = preprocess_volume(p)
        assert np.allclose(out, ref, atol=1e-5), code


def test_reverse_slices(phantom_files):
    a, _ = preprocess_volume(phantom_files["LPS"])
    b, info = preprocess_volume(phantom_files["LPS"], reverse_slices=True)
    assert info.reverse_slices and np.allclose(a[:, ::-1], b, atol=1e-6)


def test_crop_and_resize_match_monai():
    monai = pytest.importorskip("monai")
    from monai.transforms import CropForeground, Resize

    rng = np.random.default_rng(1)
    vol = np.zeros((70, 300, 280), dtype=np.float32)
    vol[8:61, 20:250, 35:270] = rng.random((53, 230, 235), dtype=np.float32) + 0.01
    vol, _ = normalize_intensity(vol, clip_percentiles=None)
    ours = resize_trilinear(crop_foreground(vol)[0], TARGET_SHAPE)
    ref = Resize(spatial_size=list(TARGET_SHAPE), mode="bilinear")(CropForeground()(vol[None]))[0]
    ref = ref.numpy() if isinstance(ref, torch.Tensor) else np.asarray(ref)
    assert ours.shape == ref.shape == TARGET_SHAPE
    assert np.abs(ours - ref).max() < 1e-5


def test_normalize_intensity():
    v = np.concatenate([np.zeros(1000), np.linspace(0, 10, 1000), [1e6]]).astype(np.float32)
    n, clip = normalize_intensity(v, (0.5, 99.5))
    assert n.min() == 0 and n.max() == 1 and clip[1] < 1e6
    n2, clip2 = normalize_intensity(v, None)
    assert clip2 is None and n2.max() == 1 and n2[1500] < 1e-4  # outlier dominates without clipping


@pytest.mark.skipif(not (UPSTREAM_REPO / "Data/data/examples/example_03.npy").exists(), reason="official examples not fetched")
def test_official_example_passthrough(tmp_path):
    p = UPSTREAM_REPO / "Data/data/examples/example_03.npy"
    vol, info = preprocess_volume(p)
    assert info.passthrough and np.array_equal(vol, np.load(p))
    out = save_montage(vol, tmp_path / "m.png")
    assert sitk.ReadImage(out).GetSize() == (8 * 256, 2 * 256)
