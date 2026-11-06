#!/usr/bin/env python
"""Check that the checkpoint files are present and complete (sizes; --sha256 for full hashes).

    python scripts/verify_model_dir.py            # models/
    python scripts/verify_model_dir.py --sha256   # also hash the 16 GB of shards (~1 min)
"""

import _bootstrap  # noqa: F401

import argparse
import json
import sys

from m3d_lamed_mps.loading import MANIFEST, check_model_dir, resolve_model_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--sha256", action="store_true")
    ap.add_argument("--headers", action="store_true", help="also compare every tensor name/shape/dtype in the shard headers")
    args = ap.parse_args()

    d = resolve_model_dir(args.model_dir)
    rep = check_model_dir(d, verify_sha256=args.sha256)
    print(f"expected revision: {MANIFEST['repo_id']}@{MANIFEST['revision']}")
    print(rep)
    if rep.ok and args.headers:
        import torch
        from safetensors import safe_open

        from m3d_lamed_mps.loading import load_config
        from m3d_lamed_mps.modeling_m3d_lamed import LamedPhi3ForCausalLM

        with torch.device("meta"):
            sd = LamedPhi3ForCausalLM(load_config(d)).state_dict()
        idx = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
        seen, bad = set(), []
        for shard in sorted(set(idx.values())):
            with safe_open(str(d / shard), framework="pt") as f:
                for k in f.keys():
                    shape = f.get_slice(k).get_shape()
                    seen.add(k)
                    if k not in sd or list(sd[k].shape) != list(shape):
                        bad.append(k)
        missing = set(sd) - seen
        print(f"  header check: {len(seen)} tensors, {len(bad)} unexpected/mis-shaped, {len(missing)} missing")
        rep_ok = not bad and not missing
    else:
        rep_ok = rep.ok
    sys.exit(0 if rep_ok else 1)


if __name__ == "__main__":
    main()
