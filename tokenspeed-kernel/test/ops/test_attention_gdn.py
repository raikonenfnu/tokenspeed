# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel import gdn_chunk_prefill


def _fla_chunk_gated_delta_rule():
    from tokenspeed_kernel.ops.attention.triton.fla.chunk import (
        chunk_gated_delta_rule,
    )

    return chunk_gated_delta_rule


def _make_inputs(*, device: str, dtype: torch.dtype, seq_len: int = 130):
    torch.manual_seed(0)
    num_q_heads = 16
    num_v_heads = 32
    head_dim = 128
    q = torch.randn(1, seq_len, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, seq_len, num_q_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, seq_len, num_v_heads, head_dim, device=device, dtype=dtype)
    beta = torch.rand(1, seq_len, num_v_heads, device=device, dtype=dtype).sigmoid()
    g = F.logsigmoid(
        torch.rand(1, seq_len, num_v_heads, device=device, dtype=torch.float32)
    )
    initial_state = (
        torch.randn(1, num_v_heads, head_dim, head_dim, device=device, dtype=dtype)
        * 0.1
    )
    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    return q, k, v, g, beta, initial_state, cu_seqlens


@pytest.mark.parametrize("solution", ["triton", "flashinfer"])
def test_gdn_chunk_prefill_matches_fla_reference(device: str, solution: str, require):
    # Each selectable backend should match the FLA reference output/state contract.
    require("attention", "gdn_chunk_prefill", solution, torch.bfloat16, "q")

    q, k, v, g, beta, initial_state, cu_seqlens = _make_inputs(
        device=device,
        dtype=torch.bfloat16,
    )
    out, final_state = gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
        qk_l2norm=True,
        output_final_state=True,
        solution=solution,
    )

    ref_out, ref_state = _fla_chunk_gated_delta_rule()(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    assert out.shape == ref_out.shape
    assert final_state.shape == ref_state.shape
    # The FLA implementation is nondeterministic due to atomics; mean error is
    # the stable signal, while max error can be noisy on a few elements.
    assert (out.float() - ref_out.float()).abs().mean() < 1e-3
    assert (final_state.float() - ref_state.float()).abs().mean() < 1e-3


@pytest.mark.parametrize("solution", ["triton", "flashinfer"])
def test_gdn_chunk_prefill_output_h_contract(device: str, solution: str, require):
    # output_h exposes backend-native checkpoint layouts used by hybrid GDN caching.
    require("attention", "gdn_chunk_prefill", solution, torch.bfloat16, "q")

    q, k, v, g, beta, initial_state, cu_seqlens = _make_inputs(
        device=device,
        dtype=torch.bfloat16,
    )
    result = gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        qk_l2norm=True,
        output_final_state=True,
        output_h=True,
        solution=solution,
    )

    if solution == "triton":
        out, final_state, h = result
        assert h.shape == (1, 3, 32, 128, 128)
    else:
        out, final_state, h, h_cu_starts = result
        assert h.shape == (2, 32, 128, 128)
        torch.testing.assert_close(h_cu_starts, torch.tensor([0, 2], device=device))

    assert out.shape == v.shape
    assert final_state.shape == initial_state.shape
