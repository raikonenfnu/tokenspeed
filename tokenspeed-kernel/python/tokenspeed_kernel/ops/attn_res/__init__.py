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
#
# Attention-Residual mixing op: RMSNorm + per-candidate softmax score + weighted
# sum over a set of residual-stream snapshots (the Kimi-K3 AttnRes block-mix).
# Dispatches to the Blackwell TMA kernel, falling back to a portable torch path
# on other hardware or for shapes outside the kernel's supported range.
import torch as _torch
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["attn_res_fwd", "attn_res_rmsnorm"]

# Blackwell kernel coverage. This must stay a subset of the authoritative
# TVM_FFI_ICHECK bounds in csrc/attn_res_binding.cu; this gate only decides when
# to fall back to torch.
_SUPPORTED_H = frozenset({4096, 5120, 6144, 7168, 8192})
_MAX_T = 16384
_MAX_N = 12


def attn_res_fwd(
    layer_residual,
    block_residual,
    res_weight,
    rms_weight,
    eps=1e-6,
    out_norm_weight=None,
    out_norm_eps=None,
):
    """Fused Attention-Residual forward.

    Candidates are ``block_residual[0..K-1]`` followed by ``layer_residual``
    (N = K + 1). Computes ``softmax_n(<RMSNorm(v_n), rms_weight * res_weight>)``
    over candidates, then the weighted sum of the raw candidates.

    Args:
        layer_residual: bf16 ``[T, H]`` current residual stream.
        block_residual: bf16 ``[K, T, H]`` K periodic snapshots.
        res_weight: bf16 ``[H]`` scorer projection weight.
        rms_weight: bf16 ``[H]`` RMSNorm weight.
        eps: RMSNorm epsilon.
        out_norm_weight: optional bf16 ``[H]``; when given, the following
            RMSNorm (same eps) is fused into the epilogue and the return value
            is the normed mix.
        out_norm_eps: Optional output RMSNorm epsilon. Defaults to ``eps``.

    Returns:
        bf16 ``[T, H]`` mixed residual (normed when ``out_norm_weight`` given).
    """
    T, H = layer_residual.shape
    if out_norm_weight is not None and Platform.get().is_amd and T > 32:
        return attn_res_rmsnorm(
            layer_residual=layer_residual,
            block_residual=block_residual.transpose(0, 1),
            res_weight=res_weight,
            score_rms_weight=rms_weight,
            score_eps=eps,
            output_rms_weight=out_norm_weight,
            output_eps=eps if out_norm_eps is None else out_norm_eps,
            num_valid_blocks=block_residual.shape[0],
        )
    N = block_residual.shape[0] + 1
    eligible = H in _SUPPORTED_H and 1 <= T <= _MAX_T and 1 <= N <= _MAX_N
    signature = format_signature(
        layer_residual=dense_tensor_format(layer_residual.dtype),
        block_residual=dense_tensor_format(block_residual.dtype),
    )
    kernel = select_kernel(
        "attn_res",
        "fwd",
        signature,
        traits={
            "fused_output_norm": out_norm_weight is not None,
            "large_prefill": T > 32,
            "hidden_size": H,
        },
        solution=None if eligible else "torch",
    )
    return kernel(
        layer_residual=layer_residual,
        block_residual=block_residual,
        res_weight=res_weight,
        rms_weight=rms_weight,
        eps=eps,
        out_norm_weight=out_norm_weight,
    )


def attn_res_rmsnorm(
    layer_residual: _torch.Tensor,
    block_residual: _torch.Tensor,
    res_weight: _torch.Tensor,
    score_rms_weight: _torch.Tensor,
    score_eps: float,
    output_rms_weight: _torch.Tensor,
    output_eps: float,
    num_valid_blocks: int,
    *,
    solution: str | None = None,
) -> _torch.Tensor:
    """Mix token-major AttnRes candidates and apply the following RMSNorm."""
    if layer_residual.dim() != 2 or block_residual.dim() != 3:
        raise ValueError(
            "expected layer [tokens, hidden] and blocks [tokens, N, hidden]"
        )
    tokens, hidden = layer_residual.shape
    if block_residual.shape[0] != tokens or block_residual.shape[2] != hidden:
        raise ValueError("block_residual token/hidden dimensions must match")
    if not 0 <= num_valid_blocks <= block_residual.shape[1]:
        raise ValueError("num_valid_blocks exceeds block-residual capacity")
    weights = (res_weight, score_rms_weight, output_rms_weight)
    if any(weight.shape != (hidden,) for weight in weights):
        raise ValueError(f"AttnRes weights must have shape ({hidden},)")
    if any(weight.device != layer_residual.device for weight in weights):
        raise ValueError("AttnRes weights must be on the input device")
    if (
        block_residual.device != layer_residual.device
        or block_residual.dtype != layer_residual.dtype
    ):
        raise ValueError("block_residual must match the input device and dtype")
    if tokens == 0:
        return _torch.empty_like(layer_residual)

    gluon_eligible = (
        layer_residual.is_cuda
        and layer_residual.dtype == _torch.bfloat16
        and 0 < tokens <= 8192
        and hidden in _SUPPORTED_H
        and num_valid_blocks + 1 <= 12
        and layer_residual.stride(1) == 1
        and block_residual.stride(2) == 1
        and all(weight.is_contiguous() for weight in weights)
        and all(weight.dtype in (_torch.bfloat16, _torch.float32) for weight in weights)
    )
    if solution is None and not gluon_eligible:
        solution = "torch"

    kernel = select_kernel(
        "attn_res",
        "rmsnorm",
        format_signature(
            layer_residual=dense_tensor_format(layer_residual.dtype),
            block_residual=dense_tensor_format(block_residual.dtype),
        ),
        traits={"hidden_size": hidden},
        solution=solution,
    )
    return kernel(
        layer_residual=layer_residual,
        block_residual=block_residual,
        res_weight=res_weight,
        score_rms_weight=score_rms_weight,
        score_eps=score_eps,
        output_rms_weight=output_rms_weight,
        output_eps=output_eps,
        num_valid_blocks=num_valid_blocks,
    )


import tokenspeed_kernel.ops.attn_res.cuda  # noqa: E402,F401

# Registration side effects (must run so select_kernel can find the backends).
import tokenspeed_kernel.ops.attn_res.gluon  # noqa: E402,F401
import tokenspeed_kernel.ops.attn_res.torch  # noqa: E402,F401
