"""AMD K3 pre-convolution KDA megafuse parity tests."""

import pytest
import torch
from tokenspeed_kernel.ops.attention import (
    kda_paged_decode,
    try_kda_fused_paged_decode,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
    causal_conv1d_update,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_amd,
    reason="AMD GPU required",
)
def test_k3_fused_paged_decode_matches_split_and_captures() -> None:
    """Preserve AMD's [V,K] state and clamped-softplus gate contracts."""
    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    heads, head_dim = 12, 128
    proj = heads * head_dim

    mixed_qkv = (torch.randn(1, 3 * proj, device=device) * 0.1).to(dtype)
    conv_weights = (torch.randn(3 * proj, 4, device=device) * 0.1).to(dtype)
    conv_seed = (torch.randn(2, 3 * proj, 3, device=device) * 0.1).to(dtype)
    f_a_out = (torch.randn(1, head_dim, device=device) * 0.1).to(dtype)
    f_b_weight = (torch.randn(proj, head_dim, device=device) * 0.1).to(dtype)
    beta = (torch.randn(1, heads, device=device) * 0.1).to(dtype)
    a_log = (torch.randn(heads, device=device) * 0.1).float()
    dt_bias = (torch.randn(heads, head_dim, device=device) * 0.1).float()
    state_seed = (
        torch.randn(2, heads, head_dim, head_dim, device=device) * 0.01
    ).float()
    read_indices = torch.tensor([1], dtype=torch.int64, device=device)
    write_indices = torch.tensor([1], dtype=torch.int64, device=device)
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32, device=device)

    split_conv = conv_seed.clone()
    split_state = state_seed.clone()
    mixed = causal_conv1d_update(
        mixed_qkv.clone(),
        split_conv,
        conv_weights,
        bias=None,
        activation="silu",
        conv_state_indices=read_indices,
        output_state_indices=write_indices.view(-1, 1),
    )
    q, k, v = torch.split(mixed, proj, dim=-1)
    g = torch.nn.functional.linear(f_a_out, f_b_weight)
    split_out = kda_paged_decode(
        q.view(1, 1, heads, head_dim),
        k.view(1, 1, heads, head_dim),
        v.view(1, 1, heads, head_dim),
        g.view(1, 1, heads, head_dim),
        beta.view(1, 1, heads),
        a_log,
        dt_bias,
        state_pool=split_state,
        read_indices=read_indices,
        write_indices=write_indices,
        cu_seqlens=cu_seqlens,
        lower_bound=-5.0,
    )

    fused_conv = conv_seed.clone()
    fused_state = state_seed.clone()

    def run_fused() -> torch.Tensor:
        out = try_kda_fused_paged_decode(
            mixed_qkv,
            conv_weights,
            fused_conv,
            f_a_out,
            f_b_weight,
            beta,
            a_log,
            dt_bias,
            state_pool=fused_state,
            read_indices=read_indices,
            write_indices=write_indices,
            num_heads=heads,
            head_dim=head_dim,
            cu_seqlens=cu_seqlens,
            lower_bound=-5.0,
        )
        assert out is not None
        return out

    fused_out = run_fused()
    torch.cuda.synchronize()
    torch.testing.assert_close(fused_out, split_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fused_conv, split_conv, atol=0, rtol=0)
    torch.testing.assert_close(fused_state, split_state, atol=5e-5, rtol=5e-5)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_fused()
    graph.replay()
    torch.cuda.synchronize()

    # The megafuse is intentionally limited to K3's TP8, single-token decode
    # geometry. A different batch must retain the registered split fallback.
    assert (
        try_kda_fused_paged_decode(
            mixed_qkv.expand(2, -1).contiguous(),
            conv_weights,
            fused_conv,
            f_a_out,
            f_b_weight,
            beta,
            a_log,
            dt_bias,
            state_pool=fused_state,
            read_indices=read_indices,
            write_indices=write_indices,
            num_heads=heads,
            head_dim=head_dim,
            cu_seqlens=cu_seqlens,
            lower_bound=-5.0,
        )
        is None
    )
