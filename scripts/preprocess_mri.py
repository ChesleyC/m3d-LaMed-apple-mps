#!/usr/bin/env python
"""Convert a 3D image (NIfTI / DICOM dir / SimpleITK format) to the M3D-LaMed input .npy (1x32x256x256, [0,1]).

    python scripts/preprocess_mri.py brain.nii.gz -o brain_m3d.npy --montage brain_m3d.png
"""

import _bootstrap  # noqa: F401

import argparse
import json

import numpy as np

from m3d_lamed_mps.preprocessing import ORIENTATIONS, preprocess_volume, save_montage


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--orientation", default="radiological", choices=ORIENTATIONS)
    ap.add_argument("--reverse-slices", action="store_true")
    ap.add_argument("--clip", type=float, nargs=2, default=(0.5, 99.5), metavar=("LO", "HI"))
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--montage", default=None)
    args = ap.parse_args()

    vol, info = preprocess_volume(args.input, orientation=args.orientation, reverse_slices=args.reverse_slices,
                                  clip_percentiles=None if args.no_clip else tuple(args.clip))
    np.save(args.output, vol)
    print(json.dumps(info.as_dict(), indent=1, default=str))
    print(f"saved {args.output}: shape {vol.shape}, range [{vol.min():.3f}, {vol.max():.3f}]")
    if args.montage:
        print("montage:", save_montage(vol, args.montage))


if __name__ == "__main__":
    main()
