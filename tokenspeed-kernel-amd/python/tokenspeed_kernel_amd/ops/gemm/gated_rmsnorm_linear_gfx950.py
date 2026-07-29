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

"""Grouped gated-RMSNorm and output projection for gfx950."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

_LANES = gl.constexpr(64)
_BLOCK_N = 16
_NUM_WARPS = 4


@gluon.jit
def _gated_rmsnorm_linear_kernel(
    recurrent_ptr,
    gate_ptr,
    norm_weight_ptr,
    projection_weight_ptr,
    output_ptr,
    eps,
    LOCAL_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    PROJECTED: gl.constexpr,
    OUTPUT_SIZE: gl.constexpr,
    GATE_KIND: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    pid_n = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(
        [1, HEAD_DIM // _LANES],
        [1, _LANES],
        [NUM_WARPS, 1],
        [1, 0],
    )
    n_layout: gl.constexpr = gl.SliceLayout(1, layout)
    d_layout: gl.constexpr = gl.SliceLayout(0, layout)
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=n_layout)
    offs_d = gl.arange(0, HEAD_DIM, layout=d_layout)
    norm_weight = gl.amd.cdna4.buffer_load(
        norm_weight_ptr,
        offs_d.to(gl.int32),
    ).to(gl.float32)
    acc = gl.zeros([BLOCK_N], gl.float32, n_layout)

    for head in range(0, LOCAL_HEADS):
        feature = head * HEAD_DIM + offs_d
        recurrent = gl.amd.cdna4.buffer_load(
            recurrent_ptr,
            feature.to(gl.int32),
        ).to(gl.float32)
        inverse_rms = gl.rsqrt(gl.sum(recurrent * recurrent, axis=0) / HEAD_DIM + eps)
        gate = gl.amd.cdna4.buffer_load(
            gate_ptr,
            feature.to(gl.int32),
        ).to(gl.float32)
        gate_activation = 1.0 / (1.0 + gl.exp(-gate))
        if GATE_KIND == 1:
            gate_activation *= gate
        # Preserve the materialized BF16 boundary before the projection.
        normalized = (recurrent * inverse_rms * norm_weight * gate_activation).to(
            gl.bfloat16
        )
        projection_weight = gl.amd.cdna4.buffer_load(
            projection_weight_ptr,
            (
                offs_n[:, None].to(gl.int64) * PROJECTED + feature[None, :].to(gl.int64)
            ).to(gl.int32),
        )
        normalized = gl.convert_layout(normalized[None, :], layout)
        acc += gl.sum(
            projection_weight.to(gl.float32) * normalized.to(gl.float32),
            axis=1,
        )

    gl.store(output_ptr + offs_n, acc.to(output_ptr.dtype.element_ty))


def gluon_gated_rmsnorm_linear_gfx950(
    recurrent: torch.Tensor,
    gate: torch.Tensor,
    norm_weight: torch.Tensor,
    projection_weight: torch.Tensor,
    *,
    eps: float,
    gate_kind: str,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply grouped gated RMSNorm and a row projection."""

    projected = int(recurrent.shape[1])
    head_dim = int(norm_weight.numel())
    if projected % head_dim:
        raise ValueError("projected width must be divisible by group width")
    local_heads = projected // head_dim
    output_size = int(projection_weight.shape[0])
    if head_dim not in {64, 128, 256} or not 1 <= local_heads <= 32:
        raise ValueError("unsupported gated RMSNorm group geometry")
    if gate_kind not in {"sigmoid", "silu"}:
        raise ValueError("gate_kind must be 'sigmoid' or 'silu'")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    expected = (
        (recurrent, (1, projected), "recurrent"),
        (gate, (1, projected), "gate"),
        (norm_weight, (head_dim,), "norm_weight"),
        (projection_weight, (output_size, projected), "projection_weight"),
    )
    for tensor, shape, name in expected:
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or not tensor.is_contiguous()
            or tensor.device != recurrent.device
        ):
            raise ValueError(
                f"{name} must be contiguous colocated BF16 with shape {shape}"
            )
    if out is None:
        out = torch.empty(
            (1, output_size),
            dtype=recurrent.dtype,
            device=recurrent.device,
        )
    elif (
        tuple(out.shape) != (1, output_size)
        or out.dtype != recurrent.dtype
        or out.device != recurrent.device
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous colocated BF16 [1, output_size]")
    block_n = 32 if projected >= 1536 else _BLOCK_N
    num_warps = _NUM_WARPS
    _gated_rmsnorm_linear_kernel[(output_size // block_n,)](
        recurrent,
        gate,
        norm_weight,
        projection_weight,
        out,
        float(eps),
        LOCAL_HEADS=local_heads,
        HEAD_DIM=head_dim,
        PROJECTED=projected,
        OUTPUT_SIZE=output_size,
        GATE_KIND=0 if gate_kind == "sigmoid" else 1,
        BLOCK_N=block_n,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
        num_stages=1,
        waves_per_eu=0,
    )
    return out


__all__ = ["gluon_gated_rmsnorm_linear_gfx950"]
