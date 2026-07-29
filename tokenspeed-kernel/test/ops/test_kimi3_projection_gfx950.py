# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

import os

import pytest
import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.gemm.kimi3 import (
    _use_gluon_largem,
    _use_gluon_mediumm,
)
from tokenspeed_kernel.ops.moe import moe_sigmoid_bias_topk


def _is_gfx950() -> bool:
    return "gfx950" in os.environ.get("PYTORCH_ROCM_ARCH", "") or (
        torch.cuda.is_available() and "MI350" in torch.cuda.get_device_name()
    )


if not _is_gfx950():
    pytest.skip("Kimi K3 projection tests require gfx950", allow_module_level=True)


@pytest.mark.parametrize(
    "input_size,output_size,num_tokens",
    [
        (7168, 3584, 1),
        (3584, 7168, 1),
        (7168, 3584, 128),
        (3584, 7168, 128),
        (7168, 3584, 768),
        (7168, 3584, 1024),
        (3584, 7168, 384),
        (3584, 7168, 512),
        (3584, 7168, 2048),
        (7168, 3584, 4096),
        (3584, 7168, 4096),
        (7168, 3584, 8192),
        (3584, 7168, 8192),
    ],
)
def test_kimi3_latent_projection_matches_torch(
    input_size: int,
    output_size: int,
    num_tokens: int,
) -> None:
    torch.manual_seed(input_size + num_tokens)
    hidden_states = torch.randn(
        num_tokens, input_size, device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn(output_size, input_size, device="cuda", dtype=torch.bfloat16)

    expected = torch.nn.functional.linear(hidden_states, weight)
    actual = tokenspeed_kernel.kimi3_latent_projection(hidden_states, weight)

    if _use_gluon_largem(num_tokens, input_size, output_size):
        assert torch.equal(actual, expected)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "m,k,n,uses_medium,uses_large",
    [
        (512, 7168, 3584, False, False),
        (768, 7168, 3584, True, False),
        (1024, 7168, 3584, True, False),
        (2048, 7168, 3584, False, False),
        (4096, 7168, 3584, False, True),
        (320, 3584, 7168, False, False),
        (384, 3584, 7168, True, False),
        (512, 3584, 7168, True, False),
        (1024, 3584, 7168, False, False),
        (2048, 3584, 7168, False, True),
    ],
)
def test_kimi3_latent_projection_dispatch_boundaries(
    m: int,
    k: int,
    n: int,
    uses_medium: bool,
    uses_large: bool,
) -> None:
    assert _use_gluon_mediumm(m, k, n) is uses_medium
    assert _use_gluon_largem(m, k, n) is uses_large


@pytest.mark.parametrize("input_size,output_size", [(7168, 3584), (3584, 7168)])
def test_kimi3_latent_projection_writes_out_and_captures(
    input_size: int,
    output_size: int,
) -> None:
    hidden_states = torch.randn(1, input_size, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(output_size, input_size, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(1, output_size, device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(hidden_states, weight)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = tokenspeed_kernel.kimi3_latent_projection(
            hidden_states, weight, out=output
        )
    assert returned.data_ptr() == output.data_ptr()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("num_tokens", [1, 2])
def test_kimi3_latent_projection_add3_matches_torch_and_captures(
    num_tokens: int,
) -> None:
    torch.manual_seed(31)
    hidden_states = torch.randn(num_tokens, 3584, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(7168, 3584, device="cuda", dtype=torch.bfloat16)
    prefix = torch.randn(num_tokens, 7168, device="cuda", dtype=torch.bfloat16)
    lane = torch.randn(num_tokens, 10752, device="cuda", dtype=torch.bfloat16)
    shared_output = lane[:, 3584:]
    expected = (
        prefix + torch.nn.functional.linear(hidden_states, weight) + shared_output
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = tokenspeed_kernel.kimi3_latent_projection_add3(
            hidden_states,
            weight,
            prefix,
            shared_output,
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_kimi3_latent_projection_rejects_forced_kernel_for_non_k3_shape() -> None:
    hidden_states = torch.empty(1, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(64, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires a supported contiguous"):
        tokenspeed_kernel.kimi3_latent_projection(
            hidden_states,
            weight,
            solution="triton_gemv",
        )


def test_kimi3_latent_projection_falls_back_for_generic_shape() -> None:
    hidden_states = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

    actual = tokenspeed_kernel.kimi3_latent_projection(hidden_states, weight)

    torch.testing.assert_close(
        actual,
        torch.nn.functional.linear(hidden_states, weight),
    )


def test_kimi3_latent_projection_add3_matches_unfused() -> None:
    torch.manual_seed(17)
    hidden_states = torch.randn(1, 3584, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(7168, 3584, device="cuda", dtype=torch.bfloat16)
    addend_a = torch.randn(1, 7168, device="cuda", dtype=torch.bfloat16)
    addend_c = torch.randn(1, 7168, device="cuda", dtype=torch.bfloat16)

    projected = tokenspeed_kernel.kimi3_latent_projection(hidden_states, weight)
    expected = (addend_a.float() + projected.float() + addend_c.float()).to(
        torch.bfloat16
    )
    actual = tokenspeed_kernel.kimi3_latent_projection_add3(
        hidden_states,
        weight,
        addend_a,
        addend_c,
    )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "local_heads,head_dim,output_size,gate_kind",
    [
        (3, 256, 7168, "sigmoid"),
        (12, 128, 7168, "sigmoid"),
        (8, 128, 4096, "silu"),
    ],
)
def test_gated_rmsnorm_linear_matches_split_and_captures(
    local_heads: int,
    head_dim: int,
    output_size: int,
    gate_kind: str,
) -> None:
    from tokenspeed_kernel.ops.gemm.triton_gemv import decode_gemv

    torch.manual_seed(41)
    projected = local_heads * head_dim
    recurrent = torch.randn(1, projected, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(1, projected, device="cuda", dtype=torch.bfloat16)
    norm_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    projection_weight = (
        torch.randn(
            output_size,
            projected,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    output = torch.empty(1, output_size, device="cuda", dtype=torch.bfloat16)
    eps = 1e-6
    recurrent_heads = recurrent.float().reshape(1, local_heads, head_dim)
    inverse_rms = torch.rsqrt(recurrent_heads.square().mean(dim=-1, keepdim=True) + eps)
    gate_heads = gate.float().reshape_as(recurrent_heads)
    gate_activation = torch.sigmoid(gate_heads)
    if gate_kind == "silu":
        gate_activation *= gate_heads
    normalized = (
        recurrent_heads * inverse_rms * norm_weight.float() * gate_activation
    ).reshape_as(recurrent)
    expected = decode_gemv(normalized.to(torch.bfloat16), projection_weight)

    tokenspeed_kernel.gated_rmsnorm_linear(
        recurrent,
        gate,
        norm_weight,
        projection_weight,
        eps=eps,
        group_size=head_dim,
        gate_kind=gate_kind,
        out=output,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = tokenspeed_kernel.gated_rmsnorm_linear(
            recurrent,
            gate,
            norm_weight,
            projection_weight,
            eps=eps,
            group_size=head_dim,
            gate_kind=gate_kind,
            out=output,
        )
    assert returned.data_ptr() == output.data_ptr()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("num_heads", [12, 16])
@pytest.mark.parametrize("num_tokens", [1, 2])
def test_mla_absorb_query_matches_split_and_captures(
    num_tokens: int,
    num_heads: int,
) -> None:
    torch.manual_seed(61)
    query = torch.randn(num_tokens, num_heads, 192, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(num_heads, 128, 512, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(
        num_tokens, num_heads, 576, device="cuda", dtype=torch.bfloat16
    )
    q_nope, q_pe = query.split((128, 64), dim=-1)
    expected = torch.empty_like(output)
    projected = torch.bmm(q_nope.transpose(0, 1).contiguous(), weight)
    expected[..., :512].copy_(projected.transpose(0, 1))
    expected[..., 512:].copy_(q_pe)

    tokenspeed_kernel.mla_absorb_query(
        q_nope,
        weight,
        query_rope=q_pe,
        out=output,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = tokenspeed_kernel.mla_absorb_query(
            q_nope,
            weight,
            query_rope=q_pe,
            out=output,
        )
    assert returned.data_ptr() == output.data_ptr()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.parametrize("output_width", [2304, 3072])
def test_mla_normalize_project_query_matches_split_and_captures(
    output_width: int,
) -> None:
    from tokenspeed_kernel.ops.gemm.triton_gemv import rowcta_gemv
    from tokenspeed_kernel.ops.layernorm.triton import rmsnorm_fused_parallel

    torch.manual_seed(66)
    query = torch.randn(1, 1536, device="cuda", dtype=torch.bfloat16)
    kv_input = torch.randn(1, 512, device="cuda", dtype=torch.bfloat16)
    kv_expected = kv_input.clone()
    kv_output = kv_input.clone()
    q_norm_weight = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
    kv_norm_weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    projection_weight = torch.randn(
        output_width, 1536, device="cuda", dtype=torch.bfloat16
    )
    q_norm = torch.empty_like(query)
    expected = torch.empty(1, output_width, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(expected)
    rmsnorm_fused_parallel(
        query,
        q_norm_weight,
        q_norm,
        kv_expected,
        kv_norm_weight,
        kv_expected,
        1e-5,
    )
    rowcta_gemv(q_norm, projection_weight, expected)

    tokenspeed_kernel.mla_normalize_project_query(
        query,
        kv_output,
        q_norm_weight,
        kv_norm_weight,
        projection_weight,
        eps=1e-5,
        out=output,
    )
    kv_output.copy_(kv_input)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = tokenspeed_kernel.mla_normalize_project_query(
            query,
            kv_output,
            q_norm_weight,
            kv_norm_weight,
            projection_weight,
            eps=1e-5,
            out=output,
        )
    assert returned.data_ptr() == output.data_ptr()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(kv_output, kv_expected, atol=0, rtol=0)
    torch.testing.assert_close(output, expected, atol=4e-3, rtol=0)


@pytest.mark.parametrize("num_heads", [12, 16])
@pytest.mark.parametrize("use_gate", [False, True])
def test_mla_project_value_matches_split_and_captures(
    num_heads: int,
    use_gate: bool,
) -> None:
    from tokenspeed_kernel.ops.activation.triton import sigmoid_mul

    torch.manual_seed(63)
    attention = torch.randn(1, num_heads, 512, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(num_heads, 512, 128, device="cuda", dtype=torch.bfloat16)
    raw_gate = torch.randn(1, num_heads * 128, device="cuda", dtype=torch.bfloat16)
    gate = raw_gate if use_gate else None
    output = torch.empty_like(raw_gate)
    projected = torch.bmm(attention.transpose(0, 1).contiguous(), weight)
    expected = projected.transpose(0, 1).reshape_as(output).contiguous()
    if gate is not None:
        sigmoid_mul(expected, gate)

    tokenspeed_kernel.mla_project_value(
        attention,
        weight,
        gate=gate,
        out=output,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = tokenspeed_kernel.mla_project_value(
            attention,
            weight,
            gate=gate,
            out=output,
        )
    assert returned.data_ptr() == output.data_ptr()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_kimi3_qkvfab_projection_matches_torch_and_captures() -> None:
    hidden_states = torch.randn(1, 7168, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(6288, 7168, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(1, 6288, device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(hidden_states, weight)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = tokenspeed_kernel.kimi3_qkvfab_projection(
            hidden_states, weight, out=output
        )
    assert returned.data_ptr() == output.data_ptr()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("num_tokens", [0, 1, 2, 4, 8, 33])
def test_kimi3_qkvfab_projection_dispatches_all_token_counts(
    num_tokens: int,
) -> None:
    hidden_states = torch.randn(num_tokens, 7168, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(6288, 7168, device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(hidden_states, weight)
    actual = tokenspeed_kernel.kimi3_qkvfab_projection(hidden_states, weight)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_kimi3_router_projection_matches_torch_and_captures() -> None:
    hidden_states = torch.randn(1, 7168, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(896, 7168, device="cuda", dtype=torch.float32) * 0.01).to(
        torch.bfloat16
    )
    correction_bias = torch.randn(896, device="cuda", dtype=torch.float32) * 0.01
    output = torch.empty(1, 896, device="cuda", dtype=torch.float32)
    expected = torch.nn.functional.linear(hidden_states.float(), weight.float())

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = tokenspeed_kernel.kimi3_router_projection(
            hidden_states, weight, out=output
        )
    assert returned.data_ptr() == output.data_ptr()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)
    expected_weights, expected_ids = moe_sigmoid_bias_topk(
        expected,
        correction_bias,
        16,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
    )
    actual_weights, actual_ids = moe_sigmoid_bias_topk(
        output,
        correction_bias,
        16,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
    )
    assert torch.equal(actual_ids, expected_ids)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("num_tokens", [0, 1, 2, 4, 8, 33])
def test_kimi3_router_projection_dispatches_all_token_counts(
    num_tokens: int,
) -> None:
    hidden_states = torch.randn(num_tokens, 7168, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(896, 7168, device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(hidden_states.float(), weight.float())
    actual = tokenspeed_kernel.kimi3_router_projection(hidden_states, weight)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("linear_beta", [None, 2.5])
def test_kimi3_shared_situ_projection_matches_reference_and_captures(
    linear_beta: float | None,
) -> None:
    torch.manual_seed(17)
    hidden_states = torch.randn(1, 7168, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1536, 7168, device="cuda", dtype=torch.bfloat16) * 0.01
    output = torch.empty(1, 768, device="cuda", dtype=torch.bfloat16)
    gate_up = torch.nn.functional.linear(hidden_states, weight)
    expected = tokenspeed_kernel.situ_and_mul(
        gate_up,
        beta=1.5,
        linear_beta=linear_beta,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = tokenspeed_kernel.kimi3_shared_situ_projection(
            hidden_states,
            weight,
            beta=1.5,
            linear_beta=linear_beta,
            out=output,
        )
    assert returned.data_ptr() == output.data_ptr()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


def test_kimi3_shared_down_projection_matches_torch_and_captures() -> None:
    torch.manual_seed(23)
    hidden_states = torch.randn(1, 768, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(7168, 768, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(1, 7168, device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(hidden_states, weight)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = tokenspeed_kernel.kimi3_shared_down_projection(
            hidden_states,
            weight,
            out=output,
        )
    assert returned.data_ptr() == output.data_ptr()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
