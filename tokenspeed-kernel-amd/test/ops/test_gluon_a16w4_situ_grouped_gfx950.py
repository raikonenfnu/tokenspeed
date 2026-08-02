"""Determinism coverage for the gfx950 grouped A16W4 SiTU MoE."""

from __future__ import annotations

import pytest
import torch

from tokenspeed_kernel_amd.ops.moe.gluon_a16w4_situ_grouped import (
    gluon_a16w4_situ_grouped_ep_gfx950,
)


def _is_gfx950() -> bool:
    return torch.cuda.is_available() and "gfx950" in getattr(
        torch.cuda.get_device_properties(0), "gcnArchName", ""
    )


if not _is_gfx950():
    pytest.skip("gfx950 is required", allow_module_level=True)


def test_grouped_decode_is_bit_exact_across_repeated_launches() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    tokens, experts, top_k = 16, 48, 8
    hidden, intermediate = 256, 256

    x = torch.randn((tokens, hidden), dtype=torch.bfloat16, device=device)
    w13 = torch.randint(
        0,
        256,
        (experts, 2 * intermediate, hidden // 2),
        dtype=torch.uint8,
        device=device,
    )
    w13_scale = torch.full(
        (experts, 2 * intermediate, hidden // 32),
        127,
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden, intermediate // 2),
        dtype=torch.uint8,
        device=device,
    )
    w2_scale = torch.full(
        (experts, hidden, intermediate // 32),
        127,
        dtype=torch.uint8,
        device=device,
    )
    topk_ids = torch.stack(
        [torch.randperm(experts, device=device)[:top_k] for _ in range(tokens)]
    ).to(torch.int32)
    topk_weights = torch.rand(
        (tokens, top_k), dtype=torch.float32, device=device
    )
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)

    def run() -> torch.Tensor:
        return gluon_a16w4_situ_grouped_ep_gfx950(
            x,
            w13,
            w13_scale,
            w2,
            w2_scale,
            topk_weights,
            topk_ids,
            situ_beta=1.0,
            situ_linear_beta=None,
        ).clone()

    expected = run()
    for _ in range(10):
        torch.testing.assert_close(run(), expected, rtol=0, atol=0)
