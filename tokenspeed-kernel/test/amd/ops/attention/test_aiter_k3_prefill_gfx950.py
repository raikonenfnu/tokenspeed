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

"""Numerical coverage for optional AITER Kimi-K3 prefill specializations."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.kda import kda_paged_prefill
from tokenspeed_kernel.ops.attention.mla import mla_prefill
from tokenspeed_kernel.ops.attention.mla.aiter import (
    aiter_mla_prefill,
    prepare_aiter_mla_prefill_plan,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.aiter import get_aiter, get_aiter_flash_kda

_PLATFORM = current_platform()
pytestmark = pytest.mark.skipif(
    not _PLATFORM.is_cdna4
    or get_aiter() is None
    or get_aiter_flash_kda() is None,
    reason="AITER Kimi-K3 prefill kernels require gfx950",
)


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = torch.linalg.vector_norm(actual.float() - expected.float())
    reference = torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    return float((error / reference).item())


def test_aiter_flashkda_matches_gluon_packed_prefill() -> None:
    torch.manual_seed(17)
    lengths = (257, 263)
    total_tokens = sum(lengths)
    heads = 12
    dim = 128
    q = torch.randn(
        (1, total_tokens, heads, dim), device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g_raw = torch.randn_like(q)
    beta_logits = torch.randn(
        (1, total_tokens, heads), device="cuda", dtype=torch.bfloat16
    )
    a_log = torch.full((heads,), -2.0, device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros((heads, dim), device="cuda", dtype=torch.float32)
    initial_state = torch.randn(
        (len(lengths), heads, dim, dim), device="cuda", dtype=torch.float32
    ) * 0.01
    cu_seqlens_cpu = torch.tensor((0, lengths[0], total_tokens), dtype=torch.int64)
    cu_seqlens = cu_seqlens_cpu.to(device="cuda")

    common = dict(
        q=q,
        k=k,
        v=v,
        g_raw=g_raw,
        beta_logits=beta_logits,
        A_log=a_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        capacity=None,
        inputs_packed=True,
        lower_bound=-5.0,
        override=None,
        recurrent_layout="v_major",
    )
    expected = kda_paged_prefill(solution="gluon", **common)
    actual = kda_paged_prefill(solution="flashkda", **common)

    assert _relative_l2(actual.out, expected.out) < 1e-2
    assert _relative_l2(actual.final_state, expected.final_state) < 5e-3
    assert torch.isfinite(actual.out).all()
    assert torch.isfinite(actual.final_state).all()


def test_aiter_mla_matches_gluon_bottom_right_causal_prefill() -> None:
    torch.manual_seed(23)
    q_len = 257
    kv_len = 385
    heads = 12
    q = torch.randn((q_len, heads, 192), device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    k = torch.randn((kv_len, heads, 192), device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    v = torch.randn((kv_len, heads, 128), device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    cu_q = torch.tensor((0, q_len), device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor((0, kv_len), device="cuda", dtype=torch.int32)
    softmax_scale = 192**-0.5
    expected = mla_prefill(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=q_len,
        max_seqlen_kv=kv_len,
        softmax_scale=softmax_scale,
        is_causal=True,
        return_lse=False,
        solution="gluon",
        out=None,
    )
    plan = prepare_aiter_mla_prefill_plan(
        q_lens_cpu=torch.tensor((q_len,), dtype=torch.int32),
        kv_lens_cpu=torch.tensor((kv_len,), dtype=torch.int32),
        num_heads=heads,
        device=q.device,
    )
    assert plan is not None
    actual = torch.empty_like(expected)
    aiter_mla_prefill(
        q=q,
        k=k,
        v=v,
        plan=plan,
        softmax_scale=softmax_scale,
        out=actual,
    )

    assert _relative_l2(actual, expected) < 3e-2
    torch.testing.assert_close(actual, expected, rtol=6e-2, atol=6e-2)
