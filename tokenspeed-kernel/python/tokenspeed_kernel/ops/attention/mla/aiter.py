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

"""AITER persistent-scheduling MLA prefill for GFX950."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.aiter import get_aiter

_HEAD_ALIGNMENT = 16
_QUERY_TILE = 256
_KV_GRANULARITY = 128


@dataclass(frozen=True)
class AiterMLAPrefillPlan:
    """Device schedule for one packed ragged causal attention invocation."""

    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    work_indptr: torch.Tensor
    work_info: torch.Tensor
    reduce_indptr: torch.Tensor
    reduce_final_map: torch.Tensor
    reduce_partial_map: torch.Tensor
    num_partial_tiles: int
    max_q_len: int
    padded_heads: int


def aiter_mla_prefill_supported() -> bool:
    """Whether the installed stack can run GFX950 FP8 MLA prefill."""
    platform = current_platform()
    return (
        platform.is_amd
        and platform.arch_version.major == 9
        and platform.arch_version.minor == 5
        and get_aiter() is not None
    )


def prepare_aiter_mla_prefill_plan(
    *,
    q_lens_cpu: torch.Tensor,
    kv_lens_cpu: torch.Tensor,
    num_heads: int,
    device: torch.device,
) -> AiterMLAPrefillPlan | None:
    """Build an AITER schedule for packed bottom-right causal attention."""
    aiter = get_aiter()
    if not aiter_mla_prefill_supported() or aiter is None:
        return None
    if q_lens_cpu.numel() == 0 or q_lens_cpu.shape != kv_lens_cpu.shape:
        return None

    q_lens_cpu = q_lens_cpu.to(device="cpu", dtype=torch.int32).contiguous()
    kv_lens_cpu = kv_lens_cpu.to(device="cpu", dtype=torch.int32).contiguous()
    padded_heads = ((num_heads + _HEAD_ALIGNMENT - 1) // _HEAD_ALIGNMENT) * _HEAD_ALIGNMENT
    max_q_len = int(q_lens_cpu.max().item())
    batch_size = q_lens_cpu.numel()

    qo_indptr_cpu = torch.zeros(batch_size + 1, dtype=torch.int32)
    kv_indptr_cpu = torch.zeros(batch_size + 1, dtype=torch.int32)
    torch.cumsum(q_lens_cpu, dim=0, out=qo_indptr_cpu[1:])
    torch.cumsum(kv_lens_cpu, dim=0, out=kv_indptr_cpu[1:])

    metadata_info = aiter.get_ps_metadata_info_v1(
        batch_size=batch_size,
        num_head_k=padded_heads,
        max_qlen=max_q_len,
        qlen_granularity=_QUERY_TILE,
    )
    metadata = [
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in metadata_info
    ]
    (
        work_metadata,
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = metadata
    aiter.get_ps_metadata_v1(
        qo_indptr_cpu,
        kv_indptr_cpu,
        kv_lens_cpu,
        1,
        padded_heads,
        work_metadata,
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        qhead_granularity=1,
        qlen_granularity=_QUERY_TILE,
        kvlen_granularity=_KV_GRANULARITY,
        block_size=1,
        is_causal=True,
    )
    num_partial_tiles = int(reduce_indptr[-1].item())
    total_kv = int(kv_indptr_cpu[-1].item())
    return AiterMLAPrefillPlan(
        qo_indptr=qo_indptr_cpu.to(device=device),
        kv_indptr=kv_indptr_cpu.to(device=device),
        kv_indices=torch.arange(total_kv, dtype=torch.int32, device=device),
        work_indptr=work_indptr,
        work_info=work_info,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
        num_partial_tiles=num_partial_tiles,
        max_q_len=max_q_len,
        padded_heads=padded_heads,
    )


_workspaces: dict[torch.device, dict[tuple[str, torch.dtype], torch.Tensor]] = {}


def _workspace(
    *,
    device: torch.device,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    buffers = _workspaces.setdefault(device, {})
    key = (name, dtype)
    required = 1
    for extent in shape:
        required *= extent
    required = max(required, 1)
    buffer = buffers.get(key)
    if buffer is None or buffer.numel() < required:
        buffer = torch.empty(required, dtype=dtype, device=device)
        buffers[key] = buffer
    return buffer[:required].view(shape if all(shape) else (required,))


def _pad_heads(tensor: torch.Tensor, padded_heads: int) -> torch.Tensor:
    heads = tensor.shape[1]
    if heads == padded_heads:
        return tensor
    repeats = (padded_heads + heads - 1) // heads
    return tensor.repeat(1, repeats, 1)[:, :padded_heads, :].contiguous()


def aiter_mla_prefill(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: AiterMLAPrefillPlan,
    softmax_scale: float,
    out: torch.Tensor,
) -> torch.Tensor:
    """Execute FP8 packed causal MLA into a caller-owned BF16 output."""
    aiter = get_aiter()
    if aiter is None:
        raise RuntimeError("AITER MLA prefill is not installed")
    if q.dtype != torch.float8_e4m3fn or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("AITER MLA prefill requires FP8 E4M3 Q, K, and V")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("Q, K, and V must have the same head count")
    if out.shape != (q.shape[0], q.shape[1], v.shape[2]):
        raise ValueError("output shape must be [total_q, heads, value_dim]")

    real_heads = q.shape[1]
    q = _pad_heads(q, plan.padded_heads)
    k = _pad_heads(k, plan.padded_heads)
    v = _pad_heads(v, plan.padded_heads)
    partial_tiles = max(plan.num_partial_tiles, 1)
    logits = _workspace(
        device=q.device,
        name="logits",
        shape=(partial_tiles * _QUERY_TILE, plan.padded_heads, v.shape[2]),
        dtype=torch.float32,
    )
    partial_lse = _workspace(
        device=q.device,
        name="partial_lse",
        shape=(partial_tiles * _QUERY_TILE, plan.padded_heads),
        dtype=torch.float32,
    )
    final_lse = _workspace(
        device=q.device,
        name="final_lse",
        shape=(q.shape[0], plan.padded_heads),
        dtype=torch.float32,
    )
    kernel_out = out
    if real_heads != plan.padded_heads:
        kernel_out = _workspace(
            device=q.device,
            name="padded_out",
            shape=(q.shape[0], plan.padded_heads, v.shape[2]),
            dtype=out.dtype,
        )
    one_scale = _workspace(
        device=q.device,
        name="one_scale",
        shape=(),
        dtype=torch.float32,
    )
    one_scale.fill_(1.0)

    aiter.mla_prefill_ps_asm_fwd(
        q,
        k,
        v,
        plan.qo_indptr,
        plan.kv_indptr,
        plan.kv_indices,
        plan.work_indptr,
        plan.work_info,
        plan.max_q_len,
        softmax_scale,
        True,
        logits,
        partial_lse,
        kernel_out,
        one_scale,
        one_scale,
        one_scale,
    )
    aiter.mla_reduce_v1(
        logits,
        partial_lse,
        plan.reduce_indptr,
        plan.reduce_final_map,
        plan.reduce_partial_map,
        _QUERY_TILE,
        0,
        kernel_out,
        final_lse,
    )
    if kernel_out is not out:
        out.copy_(kernel_out[:, :real_heads, :])
    return out
