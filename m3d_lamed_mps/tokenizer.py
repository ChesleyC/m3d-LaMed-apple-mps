# Copyright (c) 2026 Cheng Cheng (Benjamin Cheng). MIT License, see LICENSE.
# Adapted from Hugging Face Transformers 4.42.4 (Copyright (c) The HuggingFace Inc. team, Apache-2.0):
# slow LlamaTokenizer / PreTrainedTokenizer algorithm, ported onto sentencepiece (modified).
# See THIRD_PARTY_NOTICES.md.
"""Exact re-implementation of the tokenizer M3D-LaMed was trained/evaluated with.

Upstream uses ``AutoTokenizer.from_pretrained(..., use_fast=False)`` under Transformers 4.42, i.e. the
*slow* sentencepiece ``LlamaTokenizer`` with ``legacy=False`` and ``add_prefix_space=True``.
Transformers 5 no longer ships that implementation; its ``LlamaTokenizer`` agrees on the official
prompt templates but differs for inputs with repeated whitespace / newlines / tabs, a leading space,
or a literal "▁", and decodes added tokens such as ``[SEG]`` with different spacing.

This module ports the 4.42.4 slow algorithm (``LlamaTokenizer.tokenize/_tokenize``,
``PreTrainedTokenizer.tokenize`` added-token splitting, ``_decode`` and ``convert_tokens_to_string``)
onto sentencepiece directly. It is verified token-for-token against goldens produced by real
Transformers 4.42.4 (``tests/data/golden_tokenization_tf4424.json``). Dependencies: sentencepiece,
protobuf. Never executes code from the model repository.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import sentencepiece as spm
import torch
from sentencepiece import sentencepiece_model_pb2

SPIECE_UNDERLINE = "▁"


class M3DTokenizer:
    """Minimal slow-``LlamaTokenizer``-equivalent: ``__call__``, ``decode``, ``batch_decode``, ids."""

    def __init__(self, model_dir: str | os.PathLike, model_max_length: int = 512):
        d = Path(model_dir)
        proto = sentencepiece_model_pb2.ModelProto.FromString((d / "tokenizer.model").read_bytes())
        proto.normalizer_spec.add_dummy_prefix = False  # legacy=False (get_spm_processor)
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.LoadFromSerializedProto(proto.SerializeToString())

        cfg = json.loads((d / "tokenizer_config.json").read_text())
        smap = json.loads((d / "special_tokens_map.json").read_text())
        if cfg.get("legacy", True) or not cfg.get("add_prefix_space", False) or cfg.get("add_eos_token", False):
            raise ValueError("unexpected tokenizer_config: expected legacy=False, add_prefix_space=True, add_eos_token=False")
        self.add_bos_token = bool(cfg.get("add_bos_token", True))
        self.model_max_length = model_max_length
        self.padding_side = "right"

        # added tokens (id -> spec); content -> id
        self.added_tokens_decoder = {int(k): v for k, v in cfg["added_tokens_decoder"].items()}
        self.added_tokens_encoder = {v["content"]: k for k, v in self.added_tokens_decoder.items()}

        def content(x):
            return x["content"] if isinstance(x, dict) else x

        self.bos_token, self.eos_token = content(smap["bos_token"]), content(smap["eos_token"])
        self.unk_token, self.pad_token = content(smap["unk_token"]), content(smap["pad_token"])
        self.additional_special_tokens = [content(t) for t in smap.get("additional_special_tokens", [])]
        self.all_special_tokens = list(dict.fromkeys(
            [self.bos_token, self.eos_token, self.unk_token, self.pad_token] + self.additional_special_tokens))
        self.all_special_ids = {self.convert_tokens_to_ids(t) for t in self.all_special_tokens}
        self.unk_token_length = len(self.sp_model.encode(self.unk_token, out_type=str))

        vocab_size = self.sp_model.get_piece_size()
        self._legacy_added_tokens = (set(self.added_tokens_encoder) - set(self.all_special_tokens)) | {
            t for t in self.additional_special_tokens if self.convert_tokens_to_ids(t) >= vocab_size
        }
        alts = sorted(self.added_tokens_encoder, key=len, reverse=True)
        self._split_re = re.compile("(" + "|".join(re.escape(a) for a in alts) + ")")

    # ------------------------------------------------------------------ ids
    @property
    def bos_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def pad_token_id(self) -> int:
        return self.convert_tokens_to_ids(self.pad_token)

    def __len__(self) -> int:
        return max(self.sp_model.get_piece_size(), max(self.added_tokens_decoder) + 1)

    def convert_tokens_to_ids(self, token: str) -> int:
        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self.sp_model.piece_to_id(token)

    def convert_ids_to_tokens(self, ids):
        single = isinstance(ids, int)
        ids = [ids] if single else list(ids)
        out = [self.added_tokens_decoder[i]["content"] if i in self.added_tokens_decoder else self.sp_model.id_to_piece(i) for i in ids]
        return out[0] if single else out

    # ------------------------------------------------------------------ encode
    def _tokenize(self, text: str) -> list[str]:
        if not text.startswith((SPIECE_UNDERLINE, " ")):
            return self.sp_model.encode(text, out_type=str)
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        return tokens[self.unk_token_length:] if len(tokens) >= self.unk_token_length else tokens

    def tokenize(self, text: str) -> list[str]:
        if len(text) == 0:
            return []
        text = SPIECE_UNDERLINE + text.replace(SPIECE_UNDERLINE, " ")
        parts = self._split_re.split(text)
        for i, part in enumerate(parts):
            if part in self.added_tokens_encoder:
                spec = self.added_tokens_decoder[self.added_tokens_encoder[part]]
                if spec.get("rstrip") and i + 1 < len(parts) and parts[i + 1]:
                    parts[i + 1] = parts[i + 1].lstrip()
                if spec.get("lstrip") and i > 0 and parts[i - 1]:
                    parts[i - 1] = parts[i - 1].rstrip()
        tokens: list[str] = []
        for part in parts:
            if not part:
                continue
            tokens.extend([part] if part in self.added_tokens_encoder else self._tokenize(part))
        if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
            tokens = tokens[1:]
        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [self.convert_tokens_to_ids(t) for t in self.tokenize(text)]
        return ([self.bos_token_id] + ids) if (add_special_tokens and self.add_bos_token) else ids

    def __call__(self, text, return_tensors: str | None = None, add_special_tokens: bool = True):
        batch = [text] if isinstance(text, str) else list(text)
        ids = [self.encode(t, add_special_tokens=add_special_tokens) for t in batch]
        if return_tensors == "pt":
            if len({len(x) for x in ids}) != 1:
                raise ValueError("batched prompts of different lengths are not supported (no padding)")
            t = torch.tensor(ids, dtype=torch.long)
            return {"input_ids": t, "attention_mask": torch.ones_like(t)}
        if return_tensors is not None:
            raise ValueError("only return_tensors='pt' is supported")
        ids = ids[0] if isinstance(text, str) else ids
        return {"input_ids": ids}

    # ------------------------------------------------------------------ decode
    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        tokens = list(tokens)
        if tokens and tokens[0].startswith(SPIECE_UNDERLINE):
            tokens[0] = tokens[0][1:]
        current, out, prev_is_special = [], "", False
        for i, token in enumerate(tokens):
            if token in self.all_special_tokens:
                out += self.sp_model.decode(current) + token
                prev_is_special, current = True, []
            else:
                if prev_is_special and i == 1 and not token.startswith(SPIECE_UNDERLINE):
                    out += " "
                current.append(token)
                prev_is_special = False
        return out + self.sp_model.decode(current)

    def decode(self, ids, skip_special_tokens: bool = False, spaces_between_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        sub_texts, current = [], []
        for i in ids:
            if skip_special_tokens and i in self.all_special_ids:
                continue
            token = self.convert_ids_to_tokens(int(i))
            if token in self._legacy_added_tokens:
                if current:
                    s = self.convert_tokens_to_string(current)
                    if s:
                        sub_texts.append(s)
                    current = []
                sub_texts.append(token)
            else:
                current.append(token)
        if current:
            sub_texts.append(self.convert_tokens_to_string(current))
        return (" " if spaces_between_special_tokens else "").join(sub_texts)

    def batch_decode(self, sequences, **kwargs) -> list[str]:
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        return [self.decode(s, **kwargs) for s in sequences]
