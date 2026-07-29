# Copyright (c) 2026 LightSeek Foundation

import torch
from tokenspeed_kernel.ops.gemm import (
    linear_attnres_partials,
    moe_input_projections,
    rmsnorm_linear_add,
)
from tokenspeed_kernel.ops.moe import latent_moe_decode_pipeline_available


def test_moe_input_projections_portable_composition():
    torch.manual_seed(7)
    hidden = torch.randn(3, 8, dtype=torch.bfloat16)
    router = torch.randn(5, 8, dtype=torch.bfloat16)
    routed = torch.randn(4, 8, dtype=torch.bfloat16)
    shared = torch.randn(12, 8, dtype=torch.bfloat16)

    logits, routed_input, shared_input = moe_input_projections(
        hidden,
        router,
        routed,
        shared,
        gate_clamp=4.0,
        up_clamp=25.0,
    )

    raw = torch.nn.functional.linear(hidden, shared)
    gate, up = raw.chunk(2, dim=-1)
    expected_shared = (
        4.0
        * torch.tanh(gate.float() / 4.0)
        * torch.sigmoid(gate.float())
        * (25.0 * torch.tanh(up.float() / 25.0))
    ).to(hidden.dtype)
    torch.testing.assert_close(
        logits,
        torch.nn.functional.linear(hidden.float(), router.float()),
    )
    torch.testing.assert_close(
        routed_input,
        torch.nn.functional.linear(hidden, routed),
    )
    torch.testing.assert_close(shared_input, expected_shared)


def test_rmsnorm_linear_add_portable_composition():
    torch.manual_seed(11)
    hidden = torch.randn(2, 6, dtype=torch.bfloat16)
    norm_weight = torch.randn(6, dtype=torch.bfloat16)
    linear_weight = torch.randn(9, 6, dtype=torch.bfloat16)
    addend_a = torch.randn(2, 9, dtype=torch.bfloat16)
    addend_b = torch.randn(2, 9, dtype=torch.bfloat16)

    actual = rmsnorm_linear_add(
        hidden,
        norm_weight,
        linear_weight,
        addend_a,
        addend_b,
        eps=1e-5,
    )

    source = hidden.float()
    normalized = (
        source
        * torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + 1e-5)
        * norm_weight.float()
    ).to(hidden.dtype)
    projected = torch.nn.functional.linear(normalized, linear_weight)
    expected = (projected.float() + addend_a.float() + addend_b.float()).to(
        hidden.dtype
    )
    torch.testing.assert_close(actual, expected)


def test_linear_attnres_partials_portable_composition():
    torch.manual_seed(13)
    hidden = torch.randn(2, 6, dtype=torch.bfloat16)
    weight = torch.randn(9, 6, dtype=torch.bfloat16)
    blocks = torch.randn(4, 2, 6, dtype=torch.bfloat16)
    score_a = torch.randn(6, dtype=torch.bfloat16)
    score_b = torch.randn(6, dtype=torch.bfloat16)
    scratch_a = (
        torch.empty(2, dtype=torch.float32),
        torch.empty(2, dtype=torch.float32),
        torch.empty(2, 6, dtype=torch.float32),
    )
    scratch_b = tuple(torch.empty_like(tensor) for tensor in scratch_a)

    with torch.no_grad():
        actual = linear_attnres_partials(
            hidden,
            weight,
            blocks,
            score_a,
            score_b,
            scratch_a,
            scratch_b,
            eps=1e-5,
        )

    torch.testing.assert_close(
        actual,
        torch.nn.functional.linear(hidden, weight),
    )
    values = blocks.float()
    inverse_rms = torch.rsqrt(values.square().mean(dim=-1) + 1e-5)
    for score, scratch in ((score_a, scratch_a), (score_b, scratch_b)):
        logits = torch.einsum("bth,h->bt", values, score.float()) * inverse_rms
        maxima = logits.max(dim=0).values
        unnormalized = torch.exp(logits - maxima)
        torch.testing.assert_close(scratch[0], maxima)
        torch.testing.assert_close(scratch[1], unnormalized.sum(dim=0))
        torch.testing.assert_close(
            scratch[2],
            torch.einsum("bt,bth->th", unnormalized, values),
        )


def test_joint_decode_capability_rejects_incompatible_plan():
    weights = [
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((5, 8), (4, 8), (12, 8), (8, 6))
    ]
    assert not latent_moe_decode_pipeline_available(
        *weights,
        {"apply_kernel_name": "portable"},
        joint_reduce=True,
    )
