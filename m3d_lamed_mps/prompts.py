# Copyright (c) 2026 Cheng Cheng (Benjamin Cheng). MIT License, see LICENSE.
# Prompt templates copied verbatim from BAAI-DCAI/M3D (Copyright (c) 2024 BAAI-DCAI, MIT).
# See THIRD_PARTY_NOTICES.md.
"""Prompt construction exactly as in BAAI-DCAI/M3D (LaMed/src/dataset/multi_dataset.py)."""

from __future__ import annotations

from .modeling_m3d_lamed import IMAGE_TOKEN

NUM_IMAGE_TOKENS = 256

# LaMed/src/dataset/prompt_templates.py :: Caption_templates (verbatim)
CAPTION_TEMPLATES = [
    "Can you provide a caption consists of findings for this medical image?",
    "Describe the findings of the medical image you see.",
    "Please caption this medical scan with findings.",
    "What is the findings of this image?",
    "Describe this medical scan with findings.",
    "Please write a caption consists of findings for this image.",
    "Can you summarize with findings the images presented?",
    "Please caption this scan with findings.",
    "Please provide a caption consists of findings for this medical image.",
    "Can you provide a summary consists of findings of this radiograph?",
    "What are the findings presented in this medical scan?",
    "Please write a caption consists of findings for this scan.",
    "Can you provide a description consists of findings of this medical scan?",
    "Please caption this medical scan with findings.",
    "Can you provide a caption consists of findings for this medical scan?",
    "Please generate a medical report based on this image.",
    "Can you generate a diagnose report from this image.",
    "Could you analyze and provide a caption for the findings in this medical image?",
    "Please describe the observations depicted in this medical scan.",
    "Can you summarize the findings of this image in a caption?",
    "What are the significant findings in this medical image?",
    "Please provide a detailed caption outlining the findings of this image.",
    "Could you interpret and describe the findings shown in this medical scan?",
    "What conclusions can you draw from the observations in this image?",
    "Please write a descriptive caption based on the findings in this scan.",
    "What key findings can you identify from examining this medical image?",
    "Could you generate a detailed report based on the observations in this image?",
    "Can you provide a diagnosis based on the findings in this image?",
    "Please generate a comprehensive report summarizing the findings in this image.",
    "Caption the findings in this medical image?",
    "Describe the findings you see.",
    "Caption this medical scan's findings.",
    "What are the findings here?",
    "Describe these findings.",
    "Summarize the findings in these images.",
    "Caption this scan's findings.",
    "Provide a caption for this medical image's findings.",
    "Summarize the findings of this radiograph.",
    "What findings are presented in this scan?",
    "Describe this scan's findings.",
    "Generate a medical report based on this image.",
    "Can you provide a diagnosis based on this image?",
]

# The official Quickstart question.
DEFAULT_CAPTION_QUESTION = CAPTION_TEMPLATES[0]


def build_prompt(question: str, task: str = "caption", num_image_tokens: int = NUM_IMAGE_TOKENS) -> str:
    """``caption``: ``<im_patch>*256 + question`` (CapDataset, Quickstart).
    ``vqa``/``seg``/``rec``: ``<im_patch>*256 + ' ' + question`` (VQA/Seg/PosREC datasets).
    For closed-ended VQA append ``' Choices: A. .. B. .. C. .. D. ..'`` to the question yourself."""
    image_tokens = IMAGE_TOKEN * num_image_tokens
    if task == "caption":
        return image_tokens + question
    if task in ("vqa", "seg", "rec", "reg"):
        return image_tokens + " " + question
    raise ValueError(f"unknown task {task!r}")
