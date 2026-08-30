# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Correctness tests for long-context GFX950 Gluon MLA prefill."""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required for Gluon MLA tests", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx950.attention.mla.prefill import (  # noqa: E402
    gluon_mla_prefill_gfx950,
)

_HEADS = 12
_QK_DIM = 192
_V_DIM = 128
_SOFTMAX_SCALE = _QK_DIM**-0.5


@pytest.mark.parametrize("return_lse", [False, True])
def test_long_causal_split_prefill_matches_fp32_reference(
    return_lse: bool,
) -> None:
    q_len, kv_len = 128, 8192
    torch.manual_seed(19)
    q = torch.randn(q_len, _HEADS, _QK_DIM, device="cuda", dtype=torch.bfloat16) * 0.25
    k = torch.randn(kv_len, _HEADS, _QK_DIM, device="cuda", dtype=torch.bfloat16) * 0.25
    v = torch.randn(kv_len, _HEADS, _V_DIM, device="cuda", dtype=torch.bfloat16) * 0.25
    cu_q = torch.tensor([0, q_len], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_len], device="cuda", dtype=torch.int32)

    out_storage = torch.empty(
        (q_len, _HEADS, _V_DIM + 8),
        device="cuda",
        dtype=torch.bfloat16,
    )
    out_arg = out_storage[..., :_V_DIM]
    result = gluon_mla_prefill_gfx950(
        q,
        k,
        v,
        cu_q,
        cu_kv,
        q_len,
        kv_len,
        _SOFTMAX_SCALE,
        is_causal=True,
        return_lse=return_lse,
        out=out_arg,
    )
    if return_lse:
        out, lse = result
    else:
        out, lse = result, None

    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * _SOFTMAX_SCALE
    prefix = kv_len - q_len
    causal = torch.arange(kv_len, device="cuda")[None, :] <= (
        prefix + torch.arange(q_len, device="cuda")[:, None]
    )
    scores.masked_fill_(~causal[None, :, :], float("-inf"))
    ref_lse = torch.logsumexp(scores, dim=-1).transpose(0, 1)
    ref_out = torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), v.float())

    torch.testing.assert_close(out.float(), ref_out, rtol=0.02, atol=0.02)
    assert out.data_ptr() == out_arg.data_ptr()
    if lse is not None:
        torch.testing.assert_close(lse, ref_lse, rtol=2e-4, atol=2e-4)
