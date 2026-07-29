# Copyright (c) 2026 LightSeek Foundation

"""Grouped gated-RMSNorm followed by a linear projection."""

from __future__ import annotations

import torch
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def gated_rmsnorm_linear(
    x: torch.Tensor,
    gate: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    *,
    eps: float,
    group_size: int,
    gate_kind: str,
    out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Apply grouped RMSNorm, a gate, and a linear projection.

    The normalized and gated input is materialized in ``x.dtype`` before the
    projection. ``gate_kind`` is either ``"sigmoid"`` or ``"silu"``.

    Args:
        x: Input shaped ``[tokens, projected]``.
        gate: Raw gate with the same shape, dtype, and device as ``x``.
        norm_weight: Per-group RMSNorm weight shaped ``[group_size]``.
        linear_weight: Projection weight shaped ``[output_size, projected]``.
        eps: Positive RMSNorm epsilon.
        group_size: Number of features normalized together.
        gate_kind: Gate activation, ``"sigmoid"`` or ``"silu"``.
        out: Optional output shaped ``[tokens, output_size]``.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        The projected tensor shaped ``[tokens, output_size]``.
    """
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError("x must have shape [tokens, projected]")
    if gate.shape != x.shape or gate.dtype != x.dtype or gate.device != x.device:
        raise ValueError("gate must match x")
    if group_size <= 0 or x.shape[1] % group_size:
        raise ValueError("group_size must divide the projected width")
    if norm_weight.shape != (group_size,):
        raise ValueError("norm_weight must have shape [group_size]")
    if (
        norm_weight.dtype != x.dtype
        or norm_weight.device != x.device
        or linear_weight.ndim != 2
        or linear_weight.shape[1] != x.shape[1]
        or linear_weight.dtype != x.dtype
        or linear_weight.device != x.device
    ):
        raise ValueError("weights must match x dtype, device, and projected width")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if gate_kind not in {"sigmoid", "silu"}:
        raise ValueError("gate_kind must be 'sigmoid' or 'silu'")

    expected_output = (x.shape[0], linear_weight.shape[0])
    if out is None:
        out = x.new_empty(expected_output)
    elif (
        tuple(out.shape) != expected_output
        or out.dtype != x.dtype
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError(f"out must be contiguous with shape {expected_output}")

    signature = format_signature(
        x=dense_tensor_format(x.dtype),
        gate=dense_tensor_format(gate.dtype),
        norm_weight=dense_tensor_format(norm_weight.dtype),
        linear_weight=dense_tensor_format(linear_weight.dtype),
        out=dense_tensor_format(out.dtype),
    )
    traits = {
        "tokens": x.shape[0],
        "group_size": group_size,
        "num_groups": x.shape[1] // group_size,
        "projected_size": x.shape[1],
        "output_size": linear_weight.shape[0],
        "gate_kind": gate_kind,
        "inputs_contiguous": all(
            tensor.is_contiguous()
            for tensor in (x, gate, norm_weight, linear_weight, out)
        ),
    }
    try:
        kernel = select_kernel(
            "gemm",
            "gated_rmsnorm_linear",
            signature,
            traits=traits,
            solution=solution,
            override=override,
        )
    except NoKernelFoundError:
        if override is not None or solution is not None:
            raise
        kernel = None

    if kernel is not None:
        shape_params = {
            "tokens": x.shape[0],
            "group_size": group_size,
            "num_groups": x.shape[1] // group_size,
            "projected_size": x.shape[1],
            "output_size": linear_weight.shape[0],
            "gate_kind": gate_kind,
        }
        ShapeCapture.get().record(
            "gemm",
            "gated_rmsnorm_linear",
            kernel.name,
            x.dtype,
            shape_params,
        )
        with kernel_scope(
            "gemm",
            "gated_rmsnorm_linear",
            x.dtype,
            kernel_name=kernel.name,
            **shape_params,
        ):
            return kernel(
                recurrent=x,
                gate=gate,
                norm_weight=norm_weight,
                projection_weight=linear_weight,
                eps=eps,
                gate_kind=gate_kind,
                out=out,
            )

    grouped = x.float().reshape(x.shape[0], -1, group_size)
    inverse_rms = torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + eps)
    gate_fp32 = gate.float().reshape_as(grouped)
    gate_activation = torch.sigmoid(gate_fp32)
    if gate_kind == "silu":
        gate_activation = gate_fp32 * gate_activation
    normalized = (
        grouped * inverse_rms * norm_weight.float() * gate_activation
    ).reshape_as(x)
    return torch.mm(normalized.to(x.dtype), linear_weight.t(), out=out)


__all__ = ["gated_rmsnorm_linear"]
