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
