#!/usr/bin/env python
"""Print the Apple-Silicon / MPS environment and fail if it cannot run the port."""

import _bootstrap  # noqa: F401

import json
import sys

import torch

from m3d_lamed_mps.mps_utils import fmt_gib, require_mps, system_info


def main():
    info = system_info()
    print(json.dumps(info, indent=1))
    try:
        dev = require_mps()
        x = torch.randn(256, 256, device=dev, dtype=torch.bfloat16)
        (x @ x).sum().item()
    except Exception as e:  # noqa: BLE001
        print("NOT READY:", e)
        sys.exit(1)
    print(f"OK: MPS ready (bf16 matmul works), recommended GPU working set {fmt_gib(info['mps_recommended_max_memory'])}, "
          f"CPU fallback disabled")


if __name__ == "__main__":
    main()
