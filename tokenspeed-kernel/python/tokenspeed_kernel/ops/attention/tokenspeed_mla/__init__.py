# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""TokenSpeed MLA kernels exposed through tokenspeed-kernel."""

import torch
from tokenspeed_kernel.registry import error_fn


def _torch_mla_kv_pack_quantize_fp8(
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    v: torch.Tensor,
    k_scale_inv: float = 1.0,
    v_scale_inv: float = 1.0,
    k_out: torch.Tensor | None = None,
    v_out: torch.Tensor | None = None,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack MLA keys and quantize K/V when the fused package is unavailable.

    Args:
        k_nope: Non-positional keys shaped ``[tokens, heads, width]``.
        k_pe: Positional keys shaped ``[tokens, 1, width]`` or ``[tokens, width]``.
        v: Values shaped ``[tokens, heads, width]``.
        k_scale_inv: Multiplier applied to keys before conversion.
        v_scale_inv: Multiplier applied to values before conversion.
        k_out: Optional preallocated key output.
        v_out: Optional preallocated value output.
        fp8_dtype: Output FP8 dtype.
        enable_pdl: Accepted for API compatibility; Torch controls scheduling.

    Returns:
        Packed FP8 keys and FP8 values.
    """
    del enable_pdl
    if k_pe.ndim == 2:
        k_pe = k_pe.unsqueeze(1)
    k_pe = k_pe.expand(-1, k_nope.shape[1], -1)
    k = torch.cat((k_nope, k_pe), dim=-1).float().mul_(k_scale_inv).to(fp8_dtype)
    quantized_v = v.float().mul_(v_scale_inv).to(fp8_dtype)
    if k_out is not None:
        k_out.copy_(k)
        k = k_out
    if v_out is not None:
        v_out.copy_(quantized_v)
        quantized_v = v_out
    return k, quantized_v


try:
    from tokenspeed_mla import (
        get_num_sm,
        mla_kv_pack_quantize_fp8,
        tokenspeed_mla_decode,
        tokenspeed_mla_prefill,
        warmup_compile_prefill,
    )
except ImportError:
    get_num_sm = error_fn
    mla_kv_pack_quantize_fp8 = _torch_mla_kv_pack_quantize_fp8
    tokenspeed_mla_decode = error_fn
    tokenspeed_mla_prefill = error_fn
    warmup_compile_prefill = error_fn

__all__ = [
    "get_num_sm",
    "mla_kv_pack_quantize_fp8",
    "tokenspeed_mla_decode",
    "tokenspeed_mla_prefill",
    "warmup_compile_prefill",
]
