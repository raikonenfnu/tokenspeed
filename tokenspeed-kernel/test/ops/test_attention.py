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

from __future__ import annotations

import math

import pytest
import torch
from tokenspeed_kernel import (
    attn_merge_state,
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_prefill,
    mla_decode_with_kvcache,
    mla_prefill,
)
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import current_platform

platform = current_platform()
torch.manual_seed(42)

_FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz})
_INT32_MAX = 2**31 - 1


def _randn(shape: tuple[int, ...], *, device: str, dtype: torch.dtype) -> torch.Tensor:
    init_dtype = torch.bfloat16 if dtype in _FP8_DTYPES else dtype
    tensor = torch.randn(shape, device=device, dtype=init_dtype)
    if dtype != init_dtype:
        tensor = tensor.to(dtype)
    return tensor


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [(torch.bfloat16, 64, 8, 2)],
)
@pytest.mark.parametrize("solution", ["triton", "fa3", "fa4", "gluon"])
@pytest.mark.parametrize("has_sink", [False, True], ids=["no-sink", "sink"])
@pytest.mark.parametrize("is_sliding", [False, True], ids=["full", "sliding"])
def test_mha_prefill(
    device: str,
    solution: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    has_sink: bool,
    is_sliding: bool,
    require,
) -> None:
    require("attention", "mha_prefill", solution, dtype, "q")
    if solution == "fa4" and (has_sink or is_sliding):
        pytest.skip("FA4 MHA prefill does not support sinks or sliding window")

    seqlens_list = [851, 914, 1053]
    max_seqlen = max(seqlens_list)
    cu_seqlens_cpu = [0]
    for seqlen in seqlens_list:
        cu_seqlens_cpu.append(cu_seqlens_cpu[-1] + seqlen)
    seqlens = torch.tensor(seqlens_list, device=device, dtype=torch.int32)
    cu_seqlens = torch.tensor(cu_seqlens_cpu, device=device, dtype=torch.int32)
    total_tokens = int(seqlens.sum().item())

    q = _randn((total_tokens, num_q_heads, head_dim), device=device, dtype=dtype)
    k = _randn((total_tokens, num_kv_heads, head_dim), device=device, dtype=dtype)
    v = _randn((total_tokens, num_kv_heads, head_dim), device=device, dtype=dtype)
    sinks = _randn((num_q_heads,), device=device, dtype=q.dtype) if has_sink else None
    window_left = 127 if is_sliding else -1

    out = mha_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        max_seqlen=max_seqlen,
        window_left=window_left,
        sinks=sinks,
        solution=solution,
    )

    assert out.shape == q.shape
    assert not torch.isnan(out).any()


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [
        pytest.param(torch.bfloat16, 64, 8, 2, id="bf16"),
        pytest.param(torch.float8_e4m3fn, 64, 8, 2, id="fp8"),
    ],
)
@pytest.mark.parametrize("solution", ["triton", "fa3", "fa4", "flashinfer"])
def test_mha_extend_with_kvcache(
    device: str,
    solution: str,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    require,
) -> None:
    require("attention", "mha_extend_with_kvcache", solution, dtype, "q")

    batch_size = 4
    page_size = 64
    max_cache_seqlen = 256
    prefix_seqlens_list = [63, 48, 17, 80]
    query_seqlens_list = [3, 1, 2, 4]
    max_query_seqlen = max(query_seqlens_list)
    max_cache_seqlen_used = max(
        prefix_len + query_len
        for prefix_len, query_len in zip(prefix_seqlens_list, query_seqlens_list)
    )
    prefix_seqlens = torch.tensor(prefix_seqlens_list, device=device, dtype=torch.int32)
    query_seqlens = torch.tensor(query_seqlens_list, device=device, dtype=torch.int32)
    cache_seqlens = prefix_seqlens + query_seqlens
    num_blocks_per_seq = (cache_seqlens + page_size - 1) // page_size
    max_num_blocks_per_seq = (max_cache_seqlen + page_size - 1) // page_size
    total_num_blocks = int(num_blocks_per_seq.sum().item())
    total_q = int(query_seqlens.sum().item())

    q = _randn((total_q, num_q_heads, head_dim), device=device, dtype=dtype)
    cu_seqlens_q = torch.cumsum(query_seqlens, dim=0, dtype=torch.int32)
    cu_seqlens_q = torch.nn.functional.pad(cu_seqlens_q, (1, 0))
    cu_seqlens_kv = torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32)
    cu_seqlens_kv = torch.nn.functional.pad(cu_seqlens_kv, (1, 0))

    page_table = torch.zeros(
        batch_size,
        max_num_blocks_per_seq,
        device=device,
        dtype=torch.int32,
    )
    next_block = 0
    for batch_idx, num_blocks in enumerate(num_blocks_per_seq.tolist()):
        page_table[batch_idx, :num_blocks] = torch.arange(
            next_block,
            next_block + num_blocks,
            device=device,
            dtype=torch.int32,
        )
        next_block += num_blocks

    k_cache = torch.zeros(
        total_num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v_cache = torch.zeros(
        total_num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    for batch_idx, total_kv_len in enumerate(cache_seqlens.tolist()):
        num_blocks = int(num_blocks_per_seq[batch_idx].item())
        for block_idx in range(num_blocks):
            physical_block = int(page_table[batch_idx, block_idx].item())
            block_start = block_idx * page_size
            tokens_in_block = min(page_size, total_kv_len - block_start)
            if tokens_in_block > 0:
                k_cache[physical_block, :tokens_in_block] = torch.randn(
                    tokens_in_block,
                    num_kv_heads,
                    head_dim,
                    device=device,
                    dtype=torch.bfloat16 if dtype in _FP8_DTYPES else dtype,
                ).to(dtype)
                v_cache[physical_block, :tokens_in_block] = torch.randn(
                    tokens_in_block,
                    num_kv_heads,
                    head_dim,
                    device=device,
                    dtype=torch.bfloat16 if dtype in _FP8_DTYPES else dtype,
                ).to(dtype)

    out = mha_extend_with_kvcache(
        q=q,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_q=max_query_seqlen,
        max_seqlen_k=max_cache_seqlen_used,
        solution=solution,
    )

    assert out.shape == q.shape

    if solution == "triton":
        triton_out, triton_lse = mha_extend_with_kvcache(
            q=q,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=prefix_seqlens,
            max_seqlen_q=max_query_seqlen,
            max_seqlen_k=int(prefix_seqlens.max().item()),
            return_lse=True,
            solution=solution,
        )

        assert triton_out.shape == q.shape
        assert triton_lse.shape == (q.shape[0], q.shape[1])


@pytest.mark.parametrize(
    "dtype,head_dim,num_q_heads,num_kv_heads",
    [
        pytest.param(torch.bfloat16, 64, 8, 2, id="bf16"),
        pytest.param(torch.float8_e4m3fn, 64, 8, 2, id="fp8"),
    ],
)
@pytest.mark.parametrize("solution", ["triton", "fa3", "fa4", "flashinfer", "gluon"])
@pytest.mark.parametrize("seqlen_q", [1, 4], ids=["q1", "q4"])
def test_mha_decode_with_kvcache(
    device: str,
    solution: str,
    seqlen_q: int,
    dtype: torch.dtype,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    require,
) -> None:
    require("attention", "mha_decode_with_kvcache", solution, dtype, "q")

    batch_size = 4
    page_size = 64
    max_cache_seqlen = 256
    prefix_seqlens = torch.tensor([63, 129, 17, 191], device=device, dtype=torch.int32)
    cache_seqlens = prefix_seqlens + seqlen_q
    num_blocks_per_seq = (cache_seqlens + page_size - 1) // page_size
    max_num_blocks_per_seq = (max_cache_seqlen + page_size - 1) // page_size
    total_num_blocks = int(num_blocks_per_seq.sum().item())

    q = _randn(
        (batch_size * seqlen_q, num_q_heads, head_dim),
        device=device,
        dtype=dtype,
    )

    page_table = torch.zeros(
        batch_size,
        max_num_blocks_per_seq,
        device=device,
        dtype=torch.int32,
    )
    next_block = 0
    for batch_idx, num_blocks in enumerate(num_blocks_per_seq.tolist()):
        page_table[batch_idx, :num_blocks] = torch.arange(
            next_block,
            next_block + num_blocks,
            device=device,
            dtype=torch.int32,
        )
        next_block += num_blocks

    k_cache = torch.zeros(
        total_num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v_cache = torch.zeros(
        total_num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    for batch_idx, total_kv_len in enumerate(cache_seqlens.tolist()):
        num_blocks = int(num_blocks_per_seq[batch_idx].item())
        for block_idx in range(num_blocks):
            physical_block = int(page_table[batch_idx, block_idx].item())
            block_start = block_idx * page_size
            tokens_in_block = min(page_size, total_kv_len - block_start)
            if tokens_in_block > 0:
                k_cache[physical_block, :tokens_in_block] = torch.randn(
                    tokens_in_block,
                    num_kv_heads,
                    head_dim,
                    device=device,
                    dtype=torch.bfloat16 if dtype in _FP8_DTYPES else dtype,
                ).to(dtype)
                v_cache[physical_block, :tokens_in_block] = torch.randn(
                    tokens_in_block,
                    num_kv_heads,
                    head_dim,
                    device=device,
                    dtype=torch.bfloat16 if dtype in _FP8_DTYPES else dtype,
                ).to(dtype)

    out = mha_decode_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=max_cache_seqlen,
        max_seqlen_q=seqlen_q,
        solution=solution,
    )

    assert out.shape == q.shape
    assert not torch.isnan(out).any()


@pytest.mark.parametrize(
    "dtype,num_heads,qk_head_dim,v_head_dim",
    [
        pytest.param(torch.bfloat16, 128, 192, 128, id="bf16"),
        pytest.param(platform.fp8e4m3fn.dtype, 128, 192, 128, id="fp8"),
    ],
)
@pytest.mark.parametrize("solution", ["triton"])
@pytest.mark.parametrize("is_causal", [False, True], ids=["noncausal", "causal"])
def test_mla_prefill(
    device: str,
    solution: str,
    is_causal: bool,
    dtype: torch.dtype,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    require,
) -> None:
    require("attention", "mla_prefill", solution, dtype, "q")

    q_lens = [853, 1045]
    kv_lens = q_lens
    cu_seqlens_q = torch.tensor([0, 853, 1898], device=device, dtype=torch.int32)
    cu_seqlens_kv = cu_seqlens_q
    init_dtype = torch.bfloat16 if dtype in _FP8_DTYPES else dtype
    q = torch.randn(
        sum(q_lens), num_heads, qk_head_dim, device=device, dtype=init_dtype
    )
    k = torch.randn(
        sum(kv_lens), num_heads, qk_head_dim, device=device, dtype=init_dtype
    )
    v = torch.randn(
        sum(kv_lens), num_heads, v_head_dim, device=device, dtype=init_dtype
    )
    if dtype != init_dtype:
        q = q.to(dtype)
        k = k.to(dtype)
        v = v.to(dtype)
    softmax_scale = 1.0 / math.sqrt(qk_head_dim)

    out, lse = mla_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(kv_lens),
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        return_lse=True,
        solution=solution,
    )

    refs = []
    ref_lses = []
    q_offset = 0
    kv_offset = 0
    for q_len, kv_len in zip(q_lens, kv_lens, strict=True):
        q_i = q[q_offset : q_offset + q_len].float()
        k_i = k[kv_offset : kv_offset + kv_len].float()
        v_i = v[kv_offset : kv_offset + kv_len].float()
        scores = torch.einsum("qhd,khd->hqk", q_i, k_i) * softmax_scale
        if is_causal:
            q_pos = torch.arange(q_len, device=device) + max(kv_len - q_len, 0)
            k_pos = torch.arange(kv_len, device=device)
            mask = q_pos[:, None] >= k_pos[None, :]
            scores = scores.masked_fill(~mask[None, :, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        refs.append(torch.einsum("hqk,khd->qhd", probs, v_i))
        ref_lses.append(torch.logsumexp(scores, dim=-1).transpose(0, 1))
        q_offset += q_len
        kv_offset += kv_len
    out_ref = torch.cat(refs, dim=0)
    lse_ref = torch.cat(ref_lses, dim=0)

    assert out.shape == (q.shape[0], q.shape[1], v.shape[-1])
    assert lse.shape == (q.shape[0], q.shape[1])
    out_tol = 1e-1 if dtype in _FP8_DTYPES else 8e-2
    torch.testing.assert_close(out.float(), out_ref, rtol=out_tol, atol=out_tol)
    torch.testing.assert_close(lse, lse_ref, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize(
    "dtype,num_heads,kv_lora_rank,qk_rope_head_dim",
    [
        pytest.param(torch.bfloat16, 128, 512, 64, id="bf16"),
        pytest.param(platform.fp8e4m3fn.dtype, 128, 512, 64, id="fp8"),
    ],
)
@pytest.mark.parametrize("solution", ["triton"])
def test_mla_decode_with_kvcache(
    device: str,
    solution: str,
    dtype: torch.dtype,
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    require,
) -> None:
    require("attention", "mla_decode_with_kvcache", solution, dtype, "q")

    batch_size = 2
    q_len = 1
    page_size = 4
    max_seqlen_k = 7
    num_pages = 4
    qk_nope_head_dim = 128
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    init_dtype = torch.bfloat16 if dtype in _FP8_DTYPES else dtype
    q = torch.randn(
        batch_size,
        q_len,
        num_heads,
        qk_head_dim,
        device=device,
        dtype=init_dtype,
    )
    kv_cache = torch.randn(
        num_pages,
        page_size,
        1,
        qk_head_dim,
        device=device,
        dtype=init_dtype,
    )
    if dtype != init_dtype:
        q = q.to(dtype)
        kv_cache = kv_cache.to(dtype)
    page_table = torch.tensor([[0, 1], [2, 3]], device=device, dtype=torch.int32)
    cache_seqlens = torch.tensor([5, 7], device=device, dtype=torch.int32)
    softmax_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)

    out, lse = mla_decode_with_kvcache(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=max_seqlen_k,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        return_lse=True,
        solution=solution,
    )

    refs = []
    ref_lses = []
    for batch_idx in range(batch_size):
        kv_rows = []
        for pos in range(int(cache_seqlens[batch_idx].item())):
            page = page_table[batch_idx, pos // page_size]
            kv_rows.append(kv_cache[page, pos % page_size, 0])
        kv = torch.stack(kv_rows).float()
        scores = torch.einsum("hd,kd->hk", q[batch_idx, 0].float(), kv)
        scores = scores * softmax_scale
        probs = torch.softmax(scores, dim=-1)
        refs.append(torch.matmul(probs, kv[:, :kv_lora_rank]).unsqueeze(0))
        ref_lses.append(torch.logsumexp(scores, dim=-1).unsqueeze(0))
    out_ref = torch.stack(refs, dim=0)
    lse_ref = torch.stack(ref_lses, dim=0)

    assert out.shape == (batch_size, q_len, num_heads, kv_lora_rank)
    assert lse.shape == (batch_size, q_len, num_heads)
    out_tol = 1e-1 if dtype in _FP8_DTYPES else 8e-2
    torch.testing.assert_close(out.float(), out_ref, rtol=out_tol, atol=out_tol)
    torch.testing.assert_close(lse, lse_ref, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize(
    "dtype,head_dim,num_heads",
    [(torch.bfloat16, 64, 8)],
)
@pytest.mark.parametrize(
    "solution",
    ["triton", "cuda"],
)
def test_attn_merge_state(
    device: str,
    solution: str,
    dtype: torch.dtype,
    head_dim: int,
    num_heads: int,
    require,
) -> None:
    require("attention", "attn_merge_state", solution, dtype, "out_a")

    total_q = 31
    out_a = torch.randn(total_q, num_heads, head_dim, device=device, dtype=dtype)
    out_b = torch.randn(total_q, num_heads, head_dim, device=device, dtype=dtype)
    lse_a = torch.randn(total_q, num_heads, device=device, dtype=torch.float32)
    lse_b = torch.randn(total_q, num_heads, device=device, dtype=torch.float32)

    out, lse = attn_merge_state(
        out_a,
        lse_a,
        out_b,
        lse_b,
        solution=solution,
    )

    lse_ref = torch.maximum(lse_a, lse_b)
    weight_a = torch.exp(lse_a - lse_ref)
    weight_b = torch.exp(lse_b - lse_ref)
    denom = weight_a + weight_b
    out_ref = (
        out_a.float() * weight_a[..., None] + out_b.float() * weight_b[..., None]
    ) / denom[..., None]
    lse_ref = lse_ref + torch.log(denom)

    assert out.shape == out_a.shape
    assert lse.shape == lse_a.shape
    torch.testing.assert_close(out.float(), out_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# int32 -> int64 paged-KV address overflow regression
#
# Paged attention computes a KV element address roughly as::
#
#     slot = physical_page * PAGE_SIZE + page_offset
#     addr = slot * stride_buf_kbs + ...        # stride = num_kv_heads * head_dim
#
# Triton (like C) does integer math in 32 bits unless an operand is promoted to
# int64. Once ``addr`` exceeds ``INT32_MAX`` the multiply silently wraps to a
# negative address and the masked load returns zeros -> corrupted output. The
# fix casts ``physical_pages`` to int64 at the page->slot multiply (see
# ``ops/attention/triton/mha_prefill.py`` / ``mha_decode.py``). These tests guard
# that fix: a cheap isolated-arithmetic check that always runs, plus a
# memory-gated end-to-end decode regression.
# ---------------------------------------------------------------------------


@triton.jit
def _addr_kernel(
    page_table_ptr,  # int32, mirrors the real page_table dtype
    page_offsets_ptr,  # int32
    out_ptr,  # int64, receives the computed element offset
    n_elements,
    PAGE_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
    USE_INT64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Reproduce the paged-KV element-offset expression two ways.

    ``USE_INT64=True`` mirrors the fixed kernels; ``USE_INT64=False`` mirrors the
    buggy int32 arithmetic. The result is stored into an int64 buffer so the
    32-bit wrap (sign-extended on store) is observable from the host.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    physical_pages = tl.load(page_table_ptr + offs, mask=mask, other=0)
    page_offsets = tl.load(page_offsets_ptr + offs, mask=mask, other=0)

    if USE_INT64:
        # Promote at the earliest multiply; int64 propagates downstream.
        slot = physical_pages.to(tl.int64) * PAGE_SIZE + page_offsets
    else:
        # Buggy: int32 * int32 wraps before the (64-bit) pointer add.
        slot = physical_pages * PAGE_SIZE + page_offsets

    addr = slot * STRIDE
    tl.store(out_ptr + offs, addr, mask=mask)


def _run_addr_kernel(
    physical_pages: torch.Tensor,
    page_offsets: torch.Tensor,
    page_size: int,
    stride: int,
    use_int64: bool,
) -> torch.Tensor:
    n = physical_pages.numel()
    out = torch.empty(n, dtype=torch.int64, device=physical_pages.device)
    _addr_kernel[(1,)](
        physical_pages,
        page_offsets,
        out,
        n,
        PAGE_SIZE=page_size,
        STRIDE=stride,
        USE_INT64=use_int64,
        BLOCK=128,
    )
    return out


def _reference_offsets(
    physical_pages: torch.Tensor,
    page_offsets: torch.Tensor,
    page_size: int,
    stride: int,
) -> torch.Tensor:
    pp = physical_pages.to(torch.int64)
    po = page_offsets.to(torch.int64)
    return (pp * page_size + po) * stride


# (page_size, stride, physical_pages, page_offsets); stride = num_kv_heads * head_dim
_OVERFLOW_CASES = [
    # Qwen3.5 397B full-attention layer: head_dim=256, 2 KV heads -> stride 512,
    # page_size 64. ~6.4M token slots -> peak offset ~3.3e9 > INT32_MAX.
    pytest.param(64, 2 * 256, [100_000, 100_982], [0, 63], id="qwen397b_full_attn"),
    # head_dim=128, 8 KV heads -> stride 1024: overflows at smaller slot counts.
    pytest.param(64, 8 * 128, [60_000, 80_000], [1, 17], id="hd128_kv8"),
    # Exactly straddle the boundary: one slot just below, one just above 2**31.
    pytest.param(1, 512, [4_194_303, 4_194_305], [0, 0], id="straddle_boundary"),
]


@pytest.mark.parametrize("page_size,stride,pages,offsets", _OVERFLOW_CASES)
def test_paged_kv_offset_int64_vs_int32(
    device: str,
    page_size: int,
    stride: int,
    pages: list[int],
    offsets: list[int],
) -> None:
    """The int64 path matches the reference; the int32 path overflows."""
    physical_pages = torch.tensor(pages, dtype=torch.int32, device=device)
    page_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    ref = _reference_offsets(physical_pages, page_offsets, page_size, stride)

    # Sanity: the chosen config must actually exercise the > INT32_MAX regime,
    # otherwise the test would silently stop guarding anything.
    overflow_mask = ref > _INT32_MAX
    assert overflow_mask.any(), "test config does not exceed INT32_MAX"

    out_i64 = _run_addr_kernel(
        physical_pages, page_offsets, page_size, stride, use_int64=True
    )
    out_i32 = _run_addr_kernel(
        physical_pages, page_offsets, page_size, stride, use_int64=False
    )

    # Fixed path: bit-exact against the int64 reference.
    assert torch.equal(out_i64, ref), (
        f"int64 offsets diverged from reference: "
        f"{out_i64.tolist()} vs {ref.tolist()}"
    )

    # Buggy path: exposes the overflow. Where the true offset exceeds INT32_MAX
    # the int32 result is wrong, and at least one entry wraps negative.
    assert not torch.equal(out_i32, ref), (
        "int32 path unexpectedly matched the reference; the overflow this test "
        "guards against was not triggered (Triton arithmetic semantics may have "
        "changed)."
    )
    assert (out_i32[overflow_mask] != ref[overflow_mask]).all()
    assert (out_i32 < 0).any(), "expected the int32 product to wrap negative"

    # Below-threshold entries must still agree in both paths.
    safe_mask = ~overflow_mask
    if safe_mask.any():
        assert torch.equal(out_i32[safe_mask], ref[safe_mask])


def test_paged_kv_offset_no_false_positive(device: str) -> None:
    """Below INT32_MAX, int32 and int64 paths agree (guards against overcasting
    breaking small/normal cases)."""
    page_size, stride = 64, 512
    physical_pages = torch.tensor(
        [0, 1, 1000, 65_535], dtype=torch.int32, device=device
    )
    page_offsets = torch.tensor([0, 5, 63, 1], dtype=torch.int32, device=device)
    ref = _reference_offsets(physical_pages, page_offsets, page_size, stride)
    assert (ref <= _INT32_MAX).all()

    out_i64 = _run_addr_kernel(
        physical_pages, page_offsets, page_size, stride, use_int64=True
    )
    out_i32 = _run_addr_kernel(
        physical_pages, page_offsets, page_size, stride, use_int64=False
    )
    assert torch.equal(out_i64, ref)
    assert torch.equal(out_i32, ref)


def _free_gpu_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        free, _ = torch.cuda.mem_get_info()
        return int(free)
    except Exception:
        return 0


def test_decode_kvcache_high_page_offset(device: str, require) -> None:
    """Decode against KV stored past the 2**31-element boundary.

    A correct (int64) kernel reads the right page and matches a float reference.
    Reverting the ``physical_pages.to(tl.int64)`` cast makes the element offset
    wrap negative, the masked load returns zeros (or faults), and the output no
    longer matches -- failing this test.

    Needs a >4 GiB KV cache (x2 for K and V); skipped when GPU memory is tight.
    """
    require("attention", "mha_decode_with_kvcache", "triton", torch.bfloat16, "q")

    head_dim = 64
    num_q_heads = 8
    num_kv_heads = 1  # stride_buf_kbs = num_kv_heads * head_dim = 64
    page_size = 64

    # element offset at slot s is s * (num_kv_heads * head_dim).
    # Pick a physical page so slot 0 of that page already exceeds INT32_MAX.
    stride = num_kv_heads * head_dim
    high_page = (_INT32_MAX // (page_size * stride)) + 8  # comfortably past 2**31
    num_blocks = high_page + 1

    peak_offset = (high_page * page_size) * stride
    assert peak_offset > _INT32_MAX, "high_page did not cross the int32 boundary"

    bytes_per = 2  # bfloat16
    cache_elems = num_blocks * page_size * num_kv_heads * head_dim
    needed = 2 * cache_elems * bytes_per + (512 * 1024**2)  # K + V + headroom
    free = _free_gpu_bytes()
    if free < needed + (2 * 1024**3):  # keep ~2 GiB slack
        pytest.skip(
            f"needs ~{needed / 1024**3:.1f} GiB free KV cache, only "
            f"{free / 1024**3:.1f} GiB available"
        )

    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(1, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.zeros(
        num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.zeros(
        num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )

    # Populate only the high physical page (the rest stays zero / untouched).
    k_page = torch.randn(page_size, num_kv_heads, head_dim, dtype=torch.bfloat16)
    v_page = torch.randn(page_size, num_kv_heads, head_dim, dtype=torch.bfloat16)
    k_cache[high_page] = k_page.to(device)
    v_cache[high_page] = v_page.to(device)

    page_table = torch.full((1, 1), high_page, device=device, dtype=torch.int32)
    cache_seqlens = torch.tensor([page_size], device=device, dtype=torch.int32)

    out = mha_decode_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=page_size,
        max_seqlen_q=1,
        solution="triton",
    )

    # Float reference: single KV head broadcast across all query heads.
    k = k_cache[high_page, :, 0, :].float()  # [page_size, head_dim]
    v = v_cache[high_page, :, 0, :].float()
    q_f = q[0].float()  # [num_q_heads, head_dim]
    scores = (q_f @ k.t()) * scale  # [num_q_heads, page_size]
    probs = torch.softmax(scores, dim=-1)
    ref = probs @ v  # [num_q_heads, head_dim]

    out_f = out[0].float()
    assert out.shape == q.shape
    assert not torch.isnan(out_f).any()
    # Bug signature: the wrapped load returns zeros -> output collapses to ~0.
    assert out_f.abs().max() > 1e-2, "output collapsed to zero (int32 overflow?)"
    torch.testing.assert_close(out_f, ref, rtol=2e-2, atol=2e-2)
