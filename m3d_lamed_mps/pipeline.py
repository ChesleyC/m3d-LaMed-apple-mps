"""High-level MRI-to-text inference on MPS with latency breakdown and memory measurement."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from .modeling_m3d_lamed import LamedPhi3ForCausalLM
from .mps_utils import DeviceAudit, PeakMemorySampler, mps_memory, process_footprint, synchronize
from .prompts import DEFAULT_CAPTION_QUESTION, build_prompt


class _StepTimer(LogitsProcessor):
    """Called once per generated token (after its forward pass); records wall-clock timestamps."""

    def __init__(self, device: torch.device):
        self.device = device
        self.stamps: list[float] = []

    def __call__(self, input_ids, scores):
        synchronize(self.device)
        self.stamps.append(time.perf_counter())
        return scores


@dataclass
class GenerationResult:
    text: str
    output_ids: list
    question: str
    prompt_tokens: int
    new_tokens: int
    timings: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    audit: dict | None = None
    seg_logits: torch.Tensor | None = None

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "seg_logits"}
        if self.seg_logits is not None:
            d["seg_logits_shape"] = list(self.seg_logits.shape)
        return d


class M3DLaMedMPS:
    """Thin wrapper: tokenisation, image transfer, timed ``model.generate``, decoding."""

    def __init__(self, model: LamedPhi3ForCausalLM, tokenizer, device: torch.device | str = "mps"):
        self.model, self.tokenizer, self.device = model, tokenizer, torch.device(device)
        self._vision_t: list[float] = []
        # time the vision tower + projector inside generate(): stamp before the tower, after the projector
        self._hooks = [
            model.get_vision_tower().register_forward_pre_hook(self._stamp),
            model.model.mm_projector.register_forward_hook(self._stamp),
        ]

    def _stamp(self, *_):
        synchronize(self.device)
        self._vision_t.append(time.perf_counter())

    @classmethod
    def from_pretrained(cls, model_dir=None, device="mps", dtype=torch.bfloat16, attn_implementation="sdpa", verbose=True):
        from .loading import load_pretrained

        model, tok, info = load_pretrained(model_dir, device=device, dtype=dtype, attn_implementation=attn_implementation, verbose=verbose)
        obj = cls(model, tok, device)
        obj.load_info = info
        return obj

    def encode_prompt(self, question: str, task: str = "caption") -> torch.Tensor:
        prompt = build_prompt(question, task=task, num_image_tokens=self.model.num_image_tokens)
        ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        return ids.to(self.device)

    def image_to_tensor(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(image)
        if t.dim() == 4:
            t = t.unsqueeze(0)
        expected = (1, 1, *self.model.config.image_size)
        if tuple(t.shape) != expected:
            raise ValueError(f"image must be {expected[1:]} (or batched {expected}), got {tuple(t.shape)}")
        return t.to(device=self.device, dtype=self.model.dtype)

    @torch.inference_mode()
    def generate(
        self,
        image: np.ndarray | torch.Tensor,
        question: str = DEFAULT_CAPTION_QUESTION,
        task: str = "caption",
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float | None = None,
        min_new_tokens: int | None = None,
        seg: bool = False,
        audit: bool = False,
        seed: int | None = 0,
    ) -> GenerationResult:
        if seed is not None:
            torch.manual_seed(seed)
            torch.mps.manual_seed(seed) if self.device.type == "mps" else None
        input_ids = self.encode_prompt(question, task)
        image_t = self.image_to_tensor(image)
        timer = _StepTimer(self.device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, logits_processor=LogitsProcessorList([timer]),
                          pad_token_id=self.tokenizer.pad_token_id)
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)
        if min_new_tokens:
            gen_kwargs["min_new_tokens"] = min_new_tokens

        self._vision_t.clear()
        auditor = DeviceAudit() if audit else None
        synchronize(self.device)
        with PeakMemorySampler() as sampler:
            t0 = time.perf_counter()
            if auditor is not None:
                with auditor:
                    out = self.model.generate(image_t, input_ids, seg_enable=seg, **gen_kwargs)
            else:
                out = self.model.generate(image_t, input_ids, seg_enable=seg, **gen_kwargs)
            synchronize(self.device)
            t_end = time.perf_counter()
        seg_logits = None
        if seg:
            out, seg_logits = out
        ids = out[0].tolist()
        text = self.tokenizer.decode(ids, skip_special_tokens=True)

        st = timer.stamps
        ttft = (st[0] - t0) if st else None
        decode_s = (st[-1] - st[0]) if len(st) > 1 else 0.0
        vision_s = (self._vision_t[1] - self._vision_t[0]) if len(self._vision_t) >= 2 else None
        timings = {
            "total_s": round(t_end - t0, 4),
            "vision_encoder_projector_s": round(vision_s, 4) if vision_s is not None else None,
            "time_to_first_token_s": round(ttft, 4) if ttft is not None else None,
            "decode_s": round(decode_s, 4),
            "decode_tokens": max(len(st) - 1, 0),
            "decode_tokens_per_s": round((len(st) - 1) / decode_s, 2) if decode_s > 0 else None,
            "end_to_end_tokens_per_s": round(len(ids) / (t_end - t0), 2) if t_end > t0 else None,
        }
        if seg and st:
            timings["segmentation_s"] = round(t_end - st[-1], 4)
        memory = {**sampler.result(), **mps_memory(), **process_footprint()}
        return GenerationResult(
            text=text, output_ids=ids, question=question, prompt_tokens=int(input_ids.shape[1]), new_tokens=len(ids),
            timings=timings, memory=memory, audit=auditor.report.summary() if auditor else None, seg_logits=seg_logits,
        )
