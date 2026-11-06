"""Load the original M3D-LaMed-Phi-3-4B checkpoint onto MPS without executing remote code.

* never uses ``trust_remote_code`` / ``AutoModel``: the downloaded ``modeling_m3d_lamed.py`` is
  not imported; the ported classes in this package are used instead;
* the model is created on the ``meta`` device, cast to the target dtype and materialised directly
  on MPS (no 16 GB fp32 copy in memory), then every tensor of the safetensors shards is streamed
  in and copied into place;
* strict accounting: loading fails on any missing / unexpected / mis-shaped tensor, so a
  successful load proves every one of the 617 pretrained tensors was used, unchanged except for
  the dtype cast (no re-initialisation, no training).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import GenerationConfig, LlamaTokenizer
from transformers.models.phi3.modeling_phi3 import Phi3RotaryEmbedding

from .configuration_m3d_lamed import LamedPhi3Config
from .modeling_m3d_lamed import IMAGE_TOKEN, IMAGE_TOKEN_ID, SEG_TOKEN, LamedPhi3ForCausalLM
from .mps_utils import resolve_dtype, synchronize
from .tokenizer import M3DTokenizer

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DEFAULT_MODEL_DIR = PROJECT_DIR / "models"
MANIFEST = json.loads((PACKAGE_DIR / "checkpoint_manifest.json").read_text())


def display_path(path: str | os.PathLike) -> str:
    """Path relative to the project if it lies inside it (reports never record the local user's absolute paths)."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_DIR.resolve()))
    except ValueError:
        return str(path)

_KNOWN_NONPERSISTENT = {"model.rotary_emb.inv_freq", "model.rotary_emb.original_inv_freq"}


# --------------------------------------------------------------------------- model dir checks
def resolve_model_dir(model_dir: str | os.PathLike | None = None) -> Path:
    """``models/`` by default; also accepts a single sub-directory containing ``config.json``."""
    d = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    if (d / "config.json").exists():
        return d
    subs = [p.parent for p in d.glob("*/config.json")]
    return subs[0] if len(subs) == 1 else d


@dataclass
class ModelDirReport:
    model_dir: str
    present: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    size_mismatch: list = field(default_factory=list)
    sha256_mismatch: list = field(default_factory=list)
    sha256_checked: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing or self.size_mismatch or self.sha256_mismatch)

    def __str__(self) -> str:
        lines = [f"model dir: {self.model_dir}", f"  present: {len(self.present)} files"]
        for name, items in (("MISSING", self.missing), ("SIZE MISMATCH", self.size_mismatch), ("SHA256 MISMATCH", self.sha256_mismatch)):
            for it in items:
                lines.append(f"  {name}: {it}")
        if self.sha256_checked:
            lines.append(f"  sha256 verified: {len(self.sha256_checked)} files")
        lines.append("  status: " + ("OK" if self.ok else "NOT READY"))
        return "\n".join(lines)


def _sha256(path: Path, chunk: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def check_model_dir(model_dir: str | os.PathLike | None = None, require_weights: bool = True, verify_sha256: bool = False) -> ModelDirReport:
    d = resolve_model_dir(model_dir)
    rep = ModelDirReport(model_dir=display_path(d))
    for name, meta in MANIFEST["files"].items():
        if not meta.get("required", True):
            continue
        if not require_weights and name.endswith(".safetensors"):
            continue
        p = d / name
        if not p.exists():
            rep.missing.append(name)
            continue
        size = p.stat().st_size
        if size != meta["size"]:
            rep.size_mismatch.append(f"{name}: {size} bytes, expected {meta['size']} (incomplete download?)")
            continue
        rep.present.append(name)
        if verify_sha256 and "sha256" in meta:
            if _sha256(p) != meta["sha256"]:
                rep.sha256_mismatch.append(name)
            else:
                rep.sha256_checked.append(name)
    return rep


# --------------------------------------------------------------------------- config / tokenizer
def _test_mode() -> bool:
    """``M3D_LAMED_TEST_MODE=1`` (test suite only) accepts tiny synthetic checkpoints: it skips the
    identity check in :func:`load_config` and the manifest size check in :func:`load_pretrained`."""
    return os.environ.get("M3D_LAMED_TEST_MODE") == "1"


def load_config(model_dir: str | os.PathLike, check_identity: bool | None = None) -> LamedPhi3Config:
    cfg = LamedPhi3Config.from_pretrained(str(model_dir))
    if check_identity is None:
        check_identity = not _test_mode()
    expected = {"model_type": "lamed_phi3", "vision_tower": "vit3d", "mm_projector_type": "spp",
                "segmentation_module": "segvol", "hidden_size": 3072, "num_hidden_layers": 32}
    for k, v in (expected.items() if check_identity else ()):
        if getattr(cfg, k) != v:
            raise ValueError(f"unexpected config {k}={getattr(cfg, k)!r} (expected {v!r}); is this M3D-LaMed-Phi-3-4B?")
    return cfg


def load_tokenizer(model_dir: str | os.PathLike, model_max_length: int = 512, kind: str = "m3d"):
    """``kind="m3d"`` (default): :class:`M3DTokenizer`, an exact port of the Transformers-4.42 slow
    LlamaTokenizer that upstream used (``use_fast=False``). ``kind="hf"``: Transformers-5
    ``LlamaTokenizer`` (identical on the official templates, differs on unusual whitespace).
    Neither path consults the ``auto_map`` in config.json or runs repository code."""
    if kind == "m3d":
        tok = M3DTokenizer(model_dir, model_max_length=model_max_length)
    elif kind == "hf":
        tok = LlamaTokenizer.from_pretrained(str(model_dir), model_max_length=model_max_length, padding_side="right")
    else:
        raise ValueError(f"unknown tokenizer kind {kind!r}")
    for t, i in ((IMAGE_TOKEN, IMAGE_TOKEN_ID), (SEG_TOKEN, 32014)):
        if tok.convert_tokens_to_ids(t) != i:
            raise ValueError(f"tokenizer maps {t} to {tok.convert_tokens_to_ids(t)}, expected {i}")
    return tok


# --------------------------------------------------------------------------- model construction
def _reset_nonpersistent_buffers(model: LamedPhi3ForCausalLM, device: torch.device) -> None:
    sd_keys = set(model.state_dict().keys())
    nonpersistent = {n for n, _ in model.named_buffers() if n not in sd_keys}
    unknown = nonpersistent - _KNOWN_NONPERSISTENT
    if unknown:
        raise RuntimeError(f"unhandled non-persistent buffers after to_empty(): {sorted(unknown)}")
    # RoPE inverse frequencies: recomputed in float32 (as in a normal from_pretrained load);
    # they must NOT be cast to bf16/fp16.
    model.model.rotary_emb = Phi3RotaryEmbedding(config=model.config).to(device)


def build_empty_model(
    config: LamedPhi3Config,
    device: torch.device | str = "mps",
    dtype: torch.dtype | str = torch.bfloat16,
    attn_implementation: str = "sdpa",
) -> LamedPhi3ForCausalLM:
    """Allocate the model on ``device`` in ``dtype`` with *uninitialised* weights."""
    device, dtype = torch.device(device), resolve_dtype(dtype)
    with torch.device("meta"):
        model = LamedPhi3ForCausalLM(config)
    model = model.to(dtype=dtype).to_empty(device=device)
    _reset_nonpersistent_buffers(model, device)
    model.set_attn_implementation(attn_implementation)
    model.requires_grad_(False)
    return model.eval()


def init_random_weights_(model: torch.nn.Module, std: float = 0.02, seed: int = 0) -> None:
    """Synthetic weights for dry runs without the checkpoint (norms=1, biases=0, rest N(0, std))."""
    gen_device = next(model.parameters()).device
    torch.manual_seed(seed)
    if gen_device.type == "mps":
        torch.mps.manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            if "norm" in name and p.dim() >= 1 and leaf == "weight":
                p.fill_(1.0)
            elif leaf == "bias":
                p.zero_()
            else:
                p.normal_(0.0, std)
        for name, b in model.named_buffers():
            if name.endswith("positional_encoding_gaussian_matrix"):
                b.normal_(0.0, 1.0)
    synchronize(gen_device)


@dataclass
class LoadReport:
    loaded: int = 0
    expected: int = 0
    missing: list = field(default_factory=list)
    unexpected: list = field(default_factory=list)
    shape_mismatch: list = field(default_factory=list)
    source_dtypes: dict = field(default_factory=dict)
    target_dtype: str = ""
    bytes_read: int = 0
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.loaded == self.expected and not (self.missing or self.unexpected or self.shape_mismatch)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "loaded": self.loaded, "expected": self.expected, "missing": self.missing,
                "unexpected": self.unexpected, "shape_mismatch": self.shape_mismatch,
                "source_dtypes": self.source_dtypes, "target_dtype": self.target_dtype,
                "gb_read": round(self.bytes_read / 1e9, 3), "seconds": round(self.seconds, 2)}


def load_checkpoint_weights(model: torch.nn.Module, model_dir: str | os.PathLike, strict: bool = True, verbose: bool = True) -> LoadReport:
    """Stream every tensor of the sharded safetensors checkpoint into ``model`` (in place)."""
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = sorted(p.name for p in model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors found in {model_dir}")

    targets = model.state_dict(keep_vars=True)
    rep = LoadReport(expected=len(targets), target_dtype=str(next(model.parameters()).dtype).replace("torch.", ""))
    seen = set()
    t0 = time.perf_counter()
    with torch.no_grad():
        for shard in shards:
            ts = time.perf_counter()
            with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key not in targets:
                        rep.unexpected.append(key)
                        continue
                    src = f.get_tensor(key)
                    dst = targets[key]
                    if tuple(src.shape) != tuple(dst.shape):
                        rep.shape_mismatch.append(f"{key}: ckpt {tuple(src.shape)} vs model {tuple(dst.shape)}")
                        continue
                    rep.source_dtypes[str(src.dtype).replace("torch.", "")] = rep.source_dtypes.get(str(src.dtype).replace("torch.", ""), 0) + 1
                    rep.bytes_read += src.numel() * src.element_size()
                    dst.copy_(src.to(dst.dtype))  # cast on CPU first (halves host->GPU traffic)
                    seen.add(key)
            synchronize(next(model.parameters()).device)
            if verbose:
                print(f"  loaded {shard} ({time.perf_counter() - ts:.1f}s)", flush=True)
    rep.loaded = len(seen)
    rep.missing = sorted(set(targets) - seen)
    rep.seconds = time.perf_counter() - t0
    if strict and not rep.ok:
        raise RuntimeError(f"checkpoint does not match the model: {json.dumps(rep.as_dict(), indent=1)[:2000]}")
    return rep


def load_pretrained(
    model_dir: str | os.PathLike | None = None,
    device: torch.device | str = "mps",
    dtype: torch.dtype | str = torch.bfloat16,
    attn_implementation: str = "sdpa",
    verify_files: bool | None = None,
    verbose: bool = True,
):
    """Returns ``(model, tokenizer, info)`` with the original pretrained weights on ``device``."""
    d = resolve_model_dir(model_dir)
    if verify_files is None:
        verify_files = not _test_mode()
    if verify_files:
        rep = check_model_dir(d)
        if not rep.ok:
            raise FileNotFoundError(str(rep))
    t0 = time.perf_counter()
    config = load_config(d)
    tokenizer = load_tokenizer(d)
    model = build_empty_model(config, device=device, dtype=dtype, attn_implementation=attn_implementation)
    load_rep = load_checkpoint_weights(model, d, strict=True, verbose=verbose)
    model.generation_config = GenerationConfig.from_pretrained(str(d))
    synchronize(device)
    info = {"model_dir": display_path(d), "revision_expected": MANIFEST["revision"], "load": load_rep.as_dict(),
            "total_load_seconds": round(time.perf_counter() - t0, 2), "dtype": str(resolve_dtype(dtype)),
            "device": str(device), "attn_implementation": attn_implementation}
    return model, tokenizer, info
