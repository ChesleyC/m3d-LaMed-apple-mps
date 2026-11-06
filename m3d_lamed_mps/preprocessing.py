# Copyright (c) 2026 Cheng Cheng (Benjamin Cheng). MIT License, see LICENSE.
# Reproduces the BAAI-DCAI/M3D preprocessing (MIT) and the MONAI CropForeground / Resize behaviour
# (Apache-2.0). See THIRD_PARTY_NOTICES.md.
"""3D image -> M3D-LaMed input tensor ``1 x 32 x 256 x 256`` in [0, 1].

Official pipeline (BAAI-DCAI/M3D ``Data/process/m3d_cap_data_prepare.py``)::

    slices stacked on axis 0 -> /255 -> min-max to [0, 1]
    -> monai CropForeground()            (bounding box of voxels > 0, margin 0)
    -> monai Resize([32, 256, 256], mode="bilinear")   (= trilinear, align_corners=False, no anti-aliasing)

This module reproduces CropForeground + Resize exactly in NumPy/PyTorch (verified against MONAI in
tests/test_preprocessing.py) and adds what is needed for scanner-space volumes (NIfTI/DICOM/...):

* **orientation** - the volume is reoriented with SimpleITK to an axial stack ``[slice, row, col]``:
    - ``"radiological"`` (default): rows anterior->posterior, cols patient-right->left (standard
      radiological display, as the Radiopaedia images of M3D-Cap), slices inferior->superior;
    - ``"ras"``: rows posterior->anterior, cols left->right, slices inferior->superior. This is how the
      official example volumes (``Data/data/examples/*.npy``, M3D-Seg) are laid out.
* **intensity** - MRI has no fixed window, so intensities are clipped to percentiles (default
  0.5/99.5, a stand-in for the display windowing of the JPEG training images) before min-max.

Preprocessing runs on the CPU (it is data loading, not model inference).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TARGET_SHAPE = (32, 256, 256)
ORIENTATIONS = ("radiological", "ras")


@dataclass
class PreprocessInfo:
    source: str
    source_format: str
    original_shape: tuple = ()
    original_spacing: tuple = ()
    orientation: str = ""
    reverse_slices: bool = False
    clip_percentiles: tuple | None = None
    clip_values: tuple | None = None
    crop_box: tuple = ()
    cropped_shape: tuple = ()
    output_shape: tuple = TARGET_SHAPE
    passthrough: bool = False
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- reading
def _read_sitk(path: Path):
    import SimpleITK as sitk

    if path.is_dir():
        reader = sitk.ImageSeriesReader()
        series = reader.GetGDCMSeriesIDs(str(path))
        if not series:
            raise ValueError(f"no DICOM series found in {path}")
        files = max((reader.GetGDCMSeriesFileNames(str(path), s) for s in series), key=len)
        reader.SetFileNames(files)
        return reader.Execute(), "dicom"
    return sitk.ReadImage(str(path)), "".join(path.suffixes).lstrip(".") or "image"


def read_volume(path: str | os.PathLike, orientation: str = "radiological") -> tuple[np.ndarray, PreprocessInfo]:
    """Read a 3D image as float32 ``[slice, row, col]`` (axial) in the requested orientation.

    Supports NIfTI (.nii/.nii.gz), DICOM series directories, and anything SimpleITK reads
    (.mha/.nrrd/...), plus ``.npy`` arrays (assumed already ``[slice, row, col]``).
    """
    import SimpleITK as sitk

    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientation must be one of {ORIENTATIONS}")
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p).astype(np.float32)
        info = PreprocessInfo(source=str(p), source_format="npy", original_shape=tuple(arr.shape), orientation="as-stored")
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f".npy must be [D,H,W] or [1,D,H,W], got {arr.shape}")
        return arr, info

    img, fmt = _read_sitk(p)
    if img.GetDimension() == 4:
        raise ValueError(f"{p} is 4D; extract a single 3D volume first")
    if img.GetNumberOfComponentsPerPixel() != 1:
        img = sitk.VectorIndexSelectionCast(img, 0)
    info = PreprocessInfo(source=str(p), source_format=fmt, original_shape=tuple(reversed(img.GetSize())),
                          original_spacing=tuple(reversed([round(s, 4) for s in img.GetSpacing()])),
                          orientation=orientation)
    # LPS: index x -> patient Left, y -> Posterior, z -> Superior. Array is [z, y, x], i.e.
    # slices inferior->superior, rows anterior->posterior, cols right->left (= radiological display).
    img = sitk.DICOMOrient(img, "LPS")
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    if orientation == "ras":
        arr = arr[:, ::-1, ::-1]
    return np.ascontiguousarray(arr), info


# --------------------------------------------------------------------------- transforms
def normalize_intensity(vol: np.ndarray, clip_percentiles: tuple[float, float] | None = (0.5, 99.5)) -> tuple[np.ndarray, tuple | None]:
    """Optional percentile clipping, then min-max to [0, 1] (official: ``x - min; x / max(max, 1e-8)``)."""
    vol = vol.astype(np.float32, copy=True)
    clip_values = None
    if clip_percentiles is not None:
        lo, hi = np.percentile(vol, clip_percentiles)
        clip_values = (float(lo), float(hi))
        np.clip(vol, lo, hi, out=vol)
    vol -= vol.min()
    vol /= max(float(vol.max()), 1e-8)
    return vol, clip_values


def foreground_bbox(vol: np.ndarray, threshold: float = 0.0) -> tuple[tuple[int, int], ...]:
    """MONAI ``CropForeground()`` default box: voxels ``> threshold`` over all axes, margin 0."""
    mask = vol > threshold
    if not mask.any():
        return tuple((0, 0) for _ in vol.shape)
    box = []
    for ax in range(vol.ndim):
        other = tuple(a for a in range(vol.ndim) if a != ax)
        idx = np.where(mask.any(axis=other))[0]
        box.append((int(idx[0]), int(idx[-1]) + 1))
    return tuple(box)


def crop_foreground(vol: np.ndarray, threshold: float = 0.0) -> tuple[np.ndarray, tuple]:
    box = foreground_bbox(vol, threshold)
    if all(b == (0, 0) for b in box):  # MONAI returns an empty crop; keep the full volume instead
        return vol, tuple((0, s) for s in vol.shape)
    return vol[tuple(slice(a, b) for a, b in box)], box


def resize_trilinear(vol: np.ndarray, size=TARGET_SHAPE) -> np.ndarray:
    """MONAI ``Resize(spatial_size, mode="bilinear")`` on a 3D volume == ``F.interpolate(trilinear,
    align_corners=False)`` without anti-aliasing; computed in float32 on the CPU."""
    if tuple(vol.shape) == tuple(size):
        return vol.astype(np.float32, copy=False)
    t = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))[None, None]
    return F.interpolate(t, size=tuple(size), mode="trilinear", align_corners=False)[0, 0].numpy()


# --------------------------------------------------------------------------- pipeline
def preprocess_volume(
    path: str | os.PathLike,
    orientation: str = "radiological",
    reverse_slices: bool = False,
    clip_percentiles: tuple[float, float] | None = (0.5, 99.5),
) -> tuple[np.ndarray, PreprocessInfo]:
    """Full pipeline. Returns ``(array[1, 32, 256, 256] float32 in [0,1], info)``.

    A ``.npy`` that is already ``1 x 32 x 256 x 256`` in [0, 1] (e.g. the official examples) is
    passed through unchanged.
    """
    vol, info = read_volume(path, orientation=orientation)
    if info.source_format == "npy" and info.original_shape == (1, *TARGET_SHAPE) and vol.min() >= 0 and vol.max() <= 1:
        info.passthrough = True
        info.notes.append("already preprocessed (1x32x256x256 in [0,1]); passed through unchanged")
        return vol[None].astype(np.float32), info
    if reverse_slices:
        vol = vol[::-1]
        info.reverse_slices = True
    vol, info.clip_values = normalize_intensity(vol, clip_percentiles)
    info.clip_percentiles = tuple(clip_percentiles) if clip_percentiles is not None else None
    vol, box = crop_foreground(vol)
    info.crop_box, info.cropped_shape = tuple(box), tuple(vol.shape)
    vol = resize_trilinear(vol, TARGET_SHAPE)
    vol = np.clip(vol, 0.0, 1.0)  # trilinear keeps the range; guard against rounding
    return vol[None].astype(np.float32), info


def save_montage(volume: np.ndarray, path: str | os.PathLike, every: int = 2, cols: int = 8, mask: np.ndarray | None = None) -> str:
    """Write a PNG montage of slices (``[1,]D,H,W`` in [0,1]) for visual orientation QA.

    With ``mask`` (same ``[1,]D,H,W`` shape, binary) the mask is blended in red (RGB PNG).
    """
    import SimpleITK as sitk

    v = volume[0] if volume.ndim == 4 else volume
    m = None if mask is None else (mask[0] if mask.ndim == 4 else mask).astype(bool)
    if m is not None and m.shape != v.shape:
        raise ValueError(f"mask shape {m.shape} != volume shape {v.shape}")
    idx = list(range(0, v.shape[0], every))
    rows = int(np.ceil(len(idx) / cols))
    h, w = v.shape[1:]
    canvas = np.zeros((rows * h, cols * w), dtype=np.float32)
    mcanvas = np.zeros((rows * h, cols * w), dtype=bool)
    for k, i in enumerate(idx):
        r, c = divmod(k, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = v[i]
        if m is not None:
            mcanvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = m[i]
    gray = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
    if m is None:
        sitk.WriteImage(sitk.GetImageFromArray(gray), str(path))
    else:
        rgb = np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)
        rgb[mcanvas] = 0.55 * rgb[mcanvas] + 0.45 * np.array([255.0, 0.0, 0.0])
        sitk.WriteImage(sitk.GetImageFromArray(rgb.astype(np.uint8), isVector=True), str(path))
    return str(path)
