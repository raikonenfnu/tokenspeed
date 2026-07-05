import torch

from tokenspeed.runtime.layers.moe.topk import torch_native_fused_topk


def test_torch_native_fused_topk_breaks_exact_ties_by_expert_order():
    hidden_states = torch.zeros((2, 1), dtype=torch.float32)
    router_logits = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 1.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )

    topk_weights, topk_ids = torch_native_fused_topk(
        hidden_states,
        router_logits,
        topk=2,
        renormalize=True,
    )

    torch.testing.assert_close(topk_ids, torch.tensor([[0, 1], [0, 2]]))
    torch.testing.assert_close(
        topk_weights,
        torch.full((2, 2), 0.5, dtype=torch.float32),
    )


def test_torch_native_fused_topk_uses_biased_scores_only_for_choice():
    hidden_states = torch.zeros((1, 1), dtype=torch.float32)
    router_logits = torch.zeros((1, 4), dtype=torch.float32)
    correction_bias = torch.tensor([0.2, 0.1, 0.2, 0.0], dtype=torch.float32)

    topk_weights, topk_ids = torch_native_fused_topk(
        hidden_states,
        router_logits,
        topk=2,
        renormalize=True,
        correction_bias=correction_bias,
    )

    torch.testing.assert_close(topk_ids, torch.tensor([[0, 2]]))
    torch.testing.assert_close(
        topk_weights,
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
    )
