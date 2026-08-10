from types import SimpleNamespace

import pytest
import torch
from utils import is_amd, is_cdna4, is_cdna5

if not is_amd():
    pytest.skip(
        "An AMD GPU is required for MXFP4-weight Gluon MoE tests",
        allow_module_level=True,
    )


from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (  # noqa: E402
    gluon_mxfp_dynamic_mxfp4_fused_moe,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (  # noqa: E402
    gluon_mxfp_fused_moe as _gfx950_static_moe,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.situ_decode import (  # noqa: E402
    gluon_a16w4_situ_warp_decode_ep_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.situ_grouped import (  # noqa: E402
    _masked_topk_reduce_kernel,
    gluon_a16w4_situ_grouped_ep_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.weight_preprocess import (  # noqa: E402
    preprocess_gluon_mxfp4_gfx950_moe_weights,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.fused import (  # noqa: E402
    gluon_mxfp_precomputed_mxfp4_fused_moe as _gfx1250_static_moe,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.mxfp4.weight_preprocess import (  # noqa: E402
    preprocess_gluon_mxfp4_gfx1250_moe_weights,
)


def test_dynamic_mxfp4_activation_moe() -> None:
    if not is_cdna4():
        pytest.skip("Dynamic MXFP4 activation is unavailable on this GPU")

    num_tokens = 4
    num_experts = 4
    hidden_size = 256
    intermediate_size = 256
    top_k = 2
    module = torch.nn.Module()
    module.w13_input_layout = "interleaved"
    module.quant_config = SimpleNamespace(use_dynamic_mxfp4_activations=True)
    module.w13_weight = torch.nn.Parameter(
        torch.zeros(
            num_experts,
            2 * intermediate_size,
            hidden_size // 2,
            dtype=torch.uint8,
            device="cuda",
        ),
        requires_grad=False,
    )
    module.w2_weight = torch.nn.Parameter(
        torch.zeros(
            num_experts,
            hidden_size,
            intermediate_size // 2,
            dtype=torch.uint8,
            device="cuda",
        ),
        requires_grad=False,
    )
    module.w13_weight_scale = torch.nn.Parameter(
        torch.full(
            (num_experts, 2 * intermediate_size, hidden_size // 32),
            127,
            dtype=torch.uint8,
            device="cuda",
        ),
        requires_grad=False,
    )
    module.w2_weight_scale = torch.nn.Parameter(
        torch.full(
            (num_experts, hidden_size, intermediate_size // 32),
            127,
            dtype=torch.uint8,
            device="cuda",
        ),
        requires_grad=False,
    )
    preprocess_gluon_mxfp4_gfx950_moe_weights({}, module)

    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    router_logits = torch.randn(
        num_tokens, num_experts, dtype=torch.bfloat16, device="cuda"
    )
    actual = gluon_mxfp_dynamic_mxfp4_fused_moe(
        hidden_states,
        router_logits,
        module.w13_weight_triton_tensor,
        module.w2_weight_triton_tensor,
        w13_mx_scale=module.w13_precision_config.b_mx_scale,
        w2_mx_scale=module.w2_precision_config.b_mx_scale,
        top_k=top_k,
        correction_bias=None,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
    )

    torch.cuda.synchronize()
    assert actual.shape == hidden_states.shape
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)


def test_bf16_activation_situ_moe() -> None:
    if not is_cdna4():
        pytest.skip("BF16 SiTU activation is unavailable on this GPU")

    num_tokens = 1
    num_experts = 2
    hidden_size = 3584
    intermediate_size = 3072
    top_k = 2
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    w13_weight = torch.zeros(
        num_experts,
        2 * intermediate_size,
        hidden_size // 2,
        dtype=torch.uint8,
        device="cuda",
    )
    w13_scale = torch.full(
        (num_experts, 2 * intermediate_size, hidden_size // 32),
        127,
        dtype=torch.uint8,
        device="cuda",
    )
    w2_weight = torch.zeros(
        num_experts,
        hidden_size,
        intermediate_size // 2,
        dtype=torch.uint8,
        device="cuda",
    )
    w2_scale = torch.full(
        (num_experts, hidden_size, intermediate_size // 32),
        127,
        dtype=torch.uint8,
        device="cuda",
    )
    topk_weights = torch.full(
        (num_tokens, top_k), 1.0 / top_k, dtype=torch.float32, device="cuda"
    )
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")

    actual = gluon_a16w4_situ_warp_decode_ep_gfx950(
        hidden_states,
        w13_weight,
        w13_scale,
        w2_weight,
        w2_scale,
        topk_weights,
        topk_ids,
        situ_beta=4.0,
        situ_linear_beta=25.0,
        linear_weights=True,
        w13_interleaved=True,
    )

    torch.cuda.synchronize()
    assert actual.shape == hidden_states.shape
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)


def test_static_fp8_activation_moe() -> None:
    cdna4 = is_cdna4()
    if cdna4:
        hidden_size = 256
        intermediate_size = 256
        preprocess = preprocess_gluon_mxfp4_gfx950_moe_weights
    elif is_cdna5():
        hidden_size = 128
        intermediate_size = 128
        preprocess = preprocess_gluon_mxfp4_gfx1250_moe_weights
    else:
        pytest.skip("Static FP8 activation is unavailable on this GPU")

    num_tokens = 4
    num_experts = 4
    top_k = 2
    module = torch.nn.Module()
    module.w13_input_layout = "interleaved"
    module.w13_weight = torch.nn.Parameter(
        torch.zeros(
            num_experts,
            2 * intermediate_size,
            hidden_size // 2,
            dtype=torch.uint8,
            device="cuda",
        ),
        requires_grad=False,
    )
    module.w2_weight = torch.nn.Parameter(
        torch.zeros(
            num_experts,
            hidden_size,
            intermediate_size // 2,
            dtype=torch.uint8,
            device="cuda",
        ),
        requires_grad=False,
    )
    module.w13_weight_scale = torch.nn.Parameter(
        torch.full(
            (num_experts, 2 * intermediate_size, hidden_size // 32),
            127,
            dtype=torch.uint8,
            device="cuda",
        ),
        requires_grad=False,
    )
    module.w2_weight_scale = torch.nn.Parameter(
        torch.full(
            (num_experts, hidden_size, intermediate_size // 32),
            127,
            dtype=torch.uint8,
            device="cuda",
        ),
        requires_grad=False,
    )
    module.w13_input_scale = torch.nn.Parameter(
        torch.ones(1, dtype=torch.float32, device="cuda"), requires_grad=False
    )
    module.w2_input_scale = torch.nn.Parameter(
        torch.ones(1, dtype=torch.float32, device="cuda"), requires_grad=False
    )
    preprocess({}, module)

    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    if cdna4:
        router_logits = torch.randn(
            num_tokens, num_experts, dtype=torch.bfloat16, device="cuda"
        )
        actual = _gfx950_static_moe(
            hidden_states,
            router_logits,
            module.w13_weight_triton_tensor,
            module.w2_weight_triton_tensor,
            w13_mx_scale=module.w13_precision_config.b_mx_scale,
            w2_mx_scale=module.w2_precision_config.b_mx_scale,
            w13_act_scale=module.w13_act_scale,
            w2_act_scale=module.w2_act_scale,
            top_k=top_k,
        )
    else:
        topk_weights = torch.full(
            (num_tokens, top_k),
            1.0 / top_k,
            dtype=torch.float32,
            device="cuda",
        )
        topk_ids = torch.tensor(
            [[0, 1], [2, 3], [1, 2], [3, 0]], dtype=torch.int32, device="cuda"
        )
        actual = _gfx1250_static_moe(
            hidden_states,
            topk_weights,
            topk_ids,
            module.w13_weight_triton_tensor,
            module.w2_weight_triton_tensor,
            w13_mx_scale=module.w13_precision_config.b_mx_scale,
            w2_mx_scale=module.w2_precision_config.b_mx_scale,
        )

    torch.cuda.synchronize()
    assert actual.shape == hidden_states.shape
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", [5, 8, 16])
def test_grouped_decode_is_bit_exact_across_repeated_launches(
    num_tokens: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Grouped MXFP4 SiTU MoE requires an AMD GPU")
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if "gfx950" not in arch:
        pytest.skip("Grouped MXFP4 SiTU MoE is unavailable on this GPU")

    torch.manual_seed(7)
    device = torch.device("cuda")
    num_experts, top_k = 48, 8
    hidden_size, intermediate_size = 256, 256

    hidden_states = torch.randn(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device=device
    )
    w13_weight = torch.randint(
        0,
        256,
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    w13_scale = torch.full(
        (num_experts, 2 * intermediate_size, hidden_size // 32),
        127,
        dtype=torch.uint8,
        device=device,
    )
    w2_weight = torch.randint(
        0,
        256,
        (num_experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    w2_scale = torch.full(
        (num_experts, hidden_size, intermediate_size // 32),
        127,
        dtype=torch.uint8,
        device=device,
    )
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)
    topk_weights = torch.rand((num_tokens, top_k), dtype=torch.float32, device=device)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)

    def run() -> torch.Tensor:
        return gluon_a16w4_situ_grouped_ep_gfx950(
            hidden_states,
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
            topk_weights,
            topk_ids,
            situ_beta=1.0,
            situ_linear_beta=None,
        ).clone()

    expected = run()
    for _ in range(100):
        torch.testing.assert_close(run(), expected, rtol=0, atol=0)


def test_grouped_decode_reduces_slots_in_fixed_order() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Grouped MXFP4 SiTU MoE requires an AMD GPU")
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if "gfx950" not in arch:
        pytest.skip("Grouped MXFP4 SiTU MoE is unavailable on this GPU")

    device = torch.device("cuda")
    num_tokens, top_k, hidden_size = 1, 3, 256
    local_ids = torch.arange(top_k, dtype=torch.int32, device=device).view(1, top_k)
    topk_weights = torch.ones((num_tokens, top_k), dtype=torch.float32, device=device)
    terms = torch.tensor([2**24, 1, -(2**24)], dtype=torch.bfloat16, device=device)

    def reduce(partials: torch.Tensor) -> torch.Tensor:
        out = torch.empty(
            (num_tokens, hidden_size), dtype=torch.bfloat16, device=device
        )
        _masked_topk_reduce_kernel[(1,)](
            partials,
            local_ids,
            topk_weights,
            out,
            num_tokens,
            hidden_size,
            partials.stride(0),
            partials.stride(1),
            partials.stride(2),
            local_ids.stride(0),
            local_ids.stride(1),
            topk_weights.stride(0),
            topk_weights.stride(1),
            out.stride(0),
            out.stride(1),
            TOP_K=top_k,
            NUM_EXPERTS=top_k,
            EXPERT_START=0,
            BLOCK_M=64,
            BLOCK_N=hidden_size,
            num_warps=4,
        )
        return out

    # FP32 slot order is observable here: ((2**24 + 1) - 2**24) rounds to
    # zero, while ((2**24 - 2**24) + 1) is one.
    ordered = terms.view(1, top_k, 1).expand(-1, -1, hidden_size).contiguous()
    permuted = ordered[:, [0, 2, 1], :].contiguous()
    torch.testing.assert_close(
        reduce(ordered),
        torch.zeros((num_tokens, hidden_size), dtype=torch.bfloat16, device=device),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        reduce(permuted),
        torch.ones((num_tokens, hidden_size), dtype=torch.bfloat16, device=device),
        rtol=0,
        atol=0,
    )
