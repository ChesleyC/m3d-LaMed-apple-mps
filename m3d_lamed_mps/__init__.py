"""M3D-LaMed-Phi-3-4B inference on Apple Silicon (PyTorch MPS)."""

from .configuration_m3d_lamed import LamedPhi3Config
from .modeling_m3d_lamed import IMAGE_TOKEN, IMAGE_TOKEN_ID, SEG_TOKEN, LamedPhi3ForCausalLM

__all__ = ["LamedPhi3Config", "LamedPhi3ForCausalLM", "IMAGE_TOKEN", "IMAGE_TOKEN_ID", "SEG_TOKEN"]
__version__ = "0.1.0"
