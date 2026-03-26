"""
MSDGA Module - Multi-Scale Deformable And Gated Attention Operations
"""

from .ms_deform_attn import (
    MSDeformAttention,
    MSDeformAttnFunction,
    multi_scale_deformable_attn_pytorch,
    MSDeformGatedAttention,
    MSDeformGatedAttnFunction,
    multi_scale_deformable_gated_attn_pytorch,
)

__version__ = "0.1.0"
__all__ = [
    "MSDeformAttention",
    "MSDeformAttnFunction", 
    "multi_scale_deformable_attn_pytorch",
    "MSDeformGatedAttention",
    "MSDeformGatedAttnFunction", 
    "multi_scale_deformable_gated_attn_pytorch"
]