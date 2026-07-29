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

import pytest
import torch
from tokenspeed_kernel.ops.attention import mla_normalize_project_query
from tokenspeed_kernel.ops.gemm import (
    gated_rmsnorm_linear,
    kimi3_latent_projection,
    kimi3_latent_projection_add3,
    kimi3_router_projection,
    kimi3_shared_down_projection,
    kimi3_shared_situ_projection,
)


@pytest.mark.parametrize("num_tokens", [0, 3])
def test_kimi3_router_projection_has_portable_generic_fallback(
    num_tokens: int,
) -> None:
    hidden_states = torch.randn(num_tokens, 5)
    weight = torch.randn(7, 5)
    output = torch.empty(num_tokens, 7)

    returned = kimi3_router_projection(hidden_states, weight, out=output)

    assert returned.data_ptr() == output.data_ptr()
    torch.testing.assert_close(
        output, torch.nn.functional.linear(hidden_states, weight)
    )


def test_kimi3_router_forced_specialization_rejects_unsupported_inputs() -> None:
    with pytest.raises(ValueError, match="Triton router requires"):
        kimi3_router_projection(
            torch.randn(2, 5),
            torch.randn(7, 5),
            solution="triton_gemv",
        )


@pytest.mark.parametrize("num_tokens", [1, 3])
def test_kimi3_shared_projection_composes_portable_fallback(
    num_tokens: int,
) -> None:
    hidden_states = torch.randn(num_tokens, 5)
    gate_up_weight = torch.randn(8, 5)
    down_weight = torch.randn(7, 4)
    gate_up = torch.nn.functional.linear(hidden_states, gate_up_weight)
    gate, up = gate_up.chunk(2, dim=-1)
    expected_activated = torch.tanh(gate) * torch.sigmoid(gate) * up
    expected = torch.nn.functional.linear(expected_activated, down_weight)
    activated = torch.empty_like(expected_activated)
    output = torch.empty_like(expected)

    returned_activated = kimi3_shared_situ_projection(
        hidden_states,
        gate_up_weight,
        out=activated,
    )
    returned = kimi3_shared_down_projection(
        returned_activated,
        down_weight,
        out=output,
    )

    assert returned_activated.data_ptr() == activated.data_ptr()
    assert returned.data_ptr() == output.data_ptr()
    torch.testing.assert_close(activated, expected_activated)
    torch.testing.assert_close(output, expected)


def test_kimi3_latent_projection_composes_generic_fallback() -> None:
    hidden_states = torch.randn(3, 5)
    weight = torch.randn(7, 5)
    addend_a = torch.randn(3, 7)
    addend_c = torch.randn(3, 7)
    output = torch.empty(3, 7)

    projected = kimi3_latent_projection(hidden_states, weight)
    returned = kimi3_latent_projection_add3(
        hidden_states,
        weight,
        addend_a,
        addend_c,
        out=output,
    )

    expected = torch.nn.functional.linear(hidden_states, weight)
    torch.testing.assert_close(projected, expected)
    assert returned.data_ptr() == output.data_ptr()
    torch.testing.assert_close(output, addend_a + expected + addend_c)


@pytest.mark.parametrize("gate_kind", ["sigmoid", "silu"])
def test_gated_rmsnorm_linear_composes_generic_fallback(
    gate_kind: str,
) -> None:
    recurrent = torch.randn(2, 12)
    gate = torch.randn(2, 12)
    norm_weight = torch.randn(4)
    projection_weight = torch.randn(7, 12)
    eps = 1e-5

    actual = gated_rmsnorm_linear(
        recurrent,
        gate,
        norm_weight,
        projection_weight,
        eps=eps,
        group_size=4,
        gate_kind=gate_kind,
    )

    recurrent_heads = recurrent.reshape(2, 3, 4)
    normalized = recurrent_heads * torch.rsqrt(
        recurrent_heads.square().mean(-1, keepdim=True) + eps
    )
    gate_heads = gate.reshape(2, 3, 4)
    gate_activation = torch.sigmoid(gate_heads)
    if gate_kind == "silu":
        gate_activation = gate_heads * gate_activation
    normalized = (normalized * norm_weight * gate_activation).reshape(2, 12)
    expected = torch.nn.functional.linear(normalized, projection_weight)
    torch.testing.assert_close(actual, expected)


def test_mla_normalize_project_query_composes_generic_fallback() -> None:
    q = torch.randn(2, 5)
    kv = torch.randn(2, 4)[:, :3]
    assert not kv.is_contiguous()
    original_kv = kv.clone()
    q_weight = torch.randn(5)
    kv_weight = torch.randn(3)
    projection_weight = torch.randn(7, 5)
    eps = 1e-5

    actual = mla_normalize_project_query(
        q,
        kv,
        q_weight,
        kv_weight,
        projection_weight,
        eps=eps,
    )

    q_norm = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps) * q_weight
    kv_norm = (
        original_kv
        * torch.rsqrt(original_kv.square().mean(-1, keepdim=True) + eps)
        * kv_weight
    )
    torch.testing.assert_close(
        actual,
        torch.nn.functional.linear(q_norm, projection_weight),
    )
    torch.testing.assert_close(kv, kv_norm)
