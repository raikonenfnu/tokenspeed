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

"""Chunk-parallel Kimi Delta Attention prefill.

The production ``kda_recurrent`` runs a serial token-by-token scan that is
optimal for decode but leaves the machine idle during long prefills. This module
provides a chunk-parallel prefill built on the gated delta-rule (WY) chunking
scheme (Yang et al., DeltaNet-2). For each chunk the intra-chunk work is a set of
matmuls (parallel across all chunks); only the inter-chunk state carry stays
sequential:

    bg = cumsum(g)                              # per-channel gate
    A  = tril(-diag(beta) . Kd . Ki^T, -1)      # state-independent
    T  = (I - A)^{-1}                           # solve_tril
    u  = T . (beta.V) ,   W = T . (beta.Kd)     # parallel over chunks
    -- sequential over chunks --
    v_new = u - W . H
    o     = Qd . H + tril(Qd Ki^T, 0) . v_new
    H     = exp(bg_last) . H + (Kn.exp(bg_last-bg))^T . v_new

with Kd = Kn.e^{bg}, Ki = Kn.e^{-bg}, Qd = scale . Qn.e^{bg}. Chunk-local
exponentials that couple two tokens are formed as (bg_a - bg_b) with the larger
term subtracted, so the exponent stays <= 0.

Variable-length prefill: multiple requests are packed into one flat token buffer
and delimited by ``cu_seqlens`` (a prefix-sum of per-sequence lengths). Following
the repo's GDN chunk kernels, ``prepare_chunk_indices`` maps each global chunk to
its ``(sequence, local-chunk)`` so chunks never span a sequence boundary, the
chunk-local cumsum resets per sequence, and the sequential scan restarts ``H``
from each sequence's own initial state. A single sequence is just the ``N = 1``
case (``cu_seqlens = [0, T]``).

The math is validated against the serial recurrence in the kernel test suite.
The factored WY/output steps form a raw ``e^{-bg}`` term that would overflow
fp32 for long chunks, so the KKt/output kernels sub-chunk each chunk into ``BC``
rows referenced to the sub-chunk start to bound it.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

_CHUNK_SIZE = 64
_SUBCHUNK_SIZE = 16
cdna4 = gl.amd.cdna4


@gluon.jit
def _kkt_vector_fwd_kernel(
    kn,
    bg,
    beta,
    wy,
    cu_seqlens,
    chunk_indices,
    H: gl.constexpr,
    K: gl.constexpr,
    BT: gl.constexpr,
    BC: gl.constexpr,
):
    """Form the strictly-lower KDA WY matrix.

    Computes
    ``A[i,j] = beta_i * <Kn_i exp(bg_i), Kn_j exp(-bg_j)>`` for ``i > j``.
    Each ``BC``-row subchunk references both operands to its first row ``R``,
    so the row exponent ``bg_i - R`` is non-positive and the column exponent
    ``R - bg_j`` is bounded by ``BC * abs(lower_bound)``. Their product remains
    exactly ``exp(bg_i - bg_j)``.
    """
    block = gl.program_id(0)
    head = gl.program_id(1)
    row_block = block % (BT // BC)
    chunk = block // (BT // BC)

    sequence = gl.load(chunk_indices + chunk * 2).to(gl.int32)
    local_chunk = gl.load(chunk_indices + chunk * 2 + 1).to(gl.int32)
    begin = gl.load(cu_seqlens + sequence).to(gl.int32)
    end = gl.load(cu_seqlens + sequence + 1).to(gl.int32)
    length = end - begin
    row0 = local_chunk * BT + row_block * BC

    load_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [1, 1], [1, 0])
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 1],
    )
    a_layout: gl.constexpr = gl.DotOperandLayout(0, mfma_layout, k_width=8)
    b_layout: gl.constexpr = gl.DotOperandLayout(1, mfma_layout, k_width=8)
    rows = gl.arange(0, BC, layout=gl.SliceLayout(1, load_layout))
    keys = gl.arange(0, K, layout=gl.SliceLayout(0, load_layout))
    row_tokens = row0 + rows
    row_mask = row_tokens < length
    base = (begin * H + head) * K

    reference = gl.load(
        bg + base + row0 * H * K + keys,
        mask=(row0 < length) & (keys < K),
        other=0.0,
    )
    row_offsets = (base + row_tokens[:, None] * H * K + keys[None, :]).to(gl.int64)
    row_k = gl.load(
        kn + row_offsets,
        mask=row_mask[:, None] & (keys[None, :] < K),
        other=0.0,
    ).to(gl.float32)
    row_bg = gl.load(
        bg + row_offsets,
        mask=row_mask[:, None] & (keys[None, :] < K),
        other=0.0,
    ).to(gl.float32)
    row_beta = gl.load(
        beta + (begin + row_tokens) * H + head,
        mask=row_mask,
        other=0.0,
    ).to(gl.float32)
    row_k *= gl.exp(row_bg - reference[None, :]) * row_beta[:, None]

    lhs = gl.convert_layout(row_k.to(gl.bfloat16), a_layout)
    out_rows = gl.arange(0, BC, layout=gl.SliceLayout(1, mfma_layout))
    out_cols = gl.arange(0, BC, layout=gl.SliceLayout(0, mfma_layout))
    for col_block in range(row_block + 1):
        col0 = local_chunk * BT + col_block * BC
        col_tokens = col0 + rows
        col_mask = col_tokens < length
        col_offsets = (base + col_tokens[:, None] * H * K + keys[None, :]).to(gl.int64)
        col_k = gl.load(
            kn + col_offsets,
            mask=col_mask[:, None] & (keys[None, :] < K),
            other=0.0,
        ).to(gl.float32)
        col_bg = gl.load(
            bg + col_offsets,
            mask=col_mask[:, None] & (keys[None, :] < K),
            other=0.0,
        ).to(gl.float32)
        col_k *= gl.exp(reference[None, :] - col_bg)

        rhs = gl.convert_layout(col_k.trans(1, 0).to(gl.bfloat16), b_layout)
        acc = gl.zeros([BC, BC], gl.float32, mfma_layout)
        acc = cdna4.mfma(lhs, rhs, acc)
        if col_block == row_block:
            acc = gl.where(out_rows[:, None] > out_cols[None, :], acc, 0.0)
        out_offsets = (
            ((begin + row0 + out_rows[:, None]) * H + head) * BT
            + col_block * BC
            + out_cols[None, :]
        )
        gl.store(
            wy + out_offsets,
            acc,
            mask=(row0 + out_rows[:, None] < length),
        )


@gluon.jit
def _kda_prepare_gate_beta_kernel(
    raw_g,
    raw_beta,
    a_log,
    dt_bias,
    gate,
    beta,
    H: gl.constexpr,
    D: gl.constexpr,
    BLOCK_D: gl.constexpr,
    HAS_LOWER_BOUND: gl.constexpr,
    LOWER_BOUND: gl.constexpr,
):
    """Prepare the per-channel decay gate and sigmoid delta coefficient."""
    token = gl.program_id(0)
    head = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout(
        [8],
        [64],
        [gl.num_warps()],
        [0],
    )
    offsets = gl.arange(0, BLOCK_D, layout=layout)
    mask = offsets < D
    linear = (token * H + head) * D + offsets
    gate_input = gl.load(raw_g + linear, mask=mask, other=0.0).to(gl.float32)
    gate_input += gl.load(dt_bias + head * D + offsets, mask=mask, other=0.0).to(
        gl.float32
    )
    a = gl.load(a_log + head).to(gl.float32)
    if HAS_LOWER_BOUND:
        gate_value = LOWER_BOUND / (1.0 + gl.exp(-(gl.exp(a) * gate_input)))
    else:
        softplus = gl.maximum(gate_input, 0.0) + gl.log(
            1.0 + gl.exp(-gl.abs(gate_input))
        )
        gate_value = -gl.exp(a) * softplus
    gl.store(gate + linear, gate_value, mask=mask)

    beta_value = gl.load(raw_beta + token * H + head).to(gl.float32)
    gl.store(beta + token * H + head, 1.0 / (1.0 + gl.exp(-beta_value)))


@gluon.jit
def _gate_apply_kernel(
    kn,
    bg,
    beta,
    v,
    beta_kd,
    beta_v,
    kn_out,
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCK_V: gl.constexpr,
):
    """Apply the cumulative gate and beta without materializing ``exp(-bg)``.

    Emits ``beta_kd = beta * Kn * exp(bg)`` and ``beta_v = beta * V`` for the
    W/u transform. Since ``bg <= 0`` for the safe gate, the materialized
    exponential is bounded. The potentially overflowing ``exp(-bg)`` factors
    are formed from bounded differences inside the KKt and output kernels.
    """
    row = gl.program_id(0)
    key_layout: gl.constexpr = gl.BlockedLayout([8], [64], [gl.num_warps()], [0])
    key_offsets = gl.arange(0, BLOCK_K, layout=key_layout)
    beta_value = gl.load(beta + row).to(gl.float32)
    key_mask = key_offsets < K
    key_base = row * K + key_offsets
    key = gl.load(kn + key_base, mask=key_mask, other=0.0).to(gl.float32)
    cumulative_gate = gl.load(bg + key_base, mask=key_mask, other=0.0).to(gl.float32)
    gl.store(
        beta_kd + key_base,
        key * gl.exp(cumulative_gate) * beta_value,
        mask=key_mask,
    )
    gl.store(kn_out + key_base, key.to(gl.bfloat16), mask=key_mask)

    value_layout: gl.constexpr = gl.BlockedLayout([8], [64], [gl.num_warps()], [0])
    value_offsets = gl.arange(0, BLOCK_V, layout=value_layout)
    value_mask = value_offsets < V
    value_base = row * V + value_offsets
    value = gl.load(v + value_base, mask=value_mask, other=0.0).to(gl.float32)
    gl.store(
        beta_v + value_base,
        (value * beta_value).to(gl.bfloat16),
        mask=value_mask,
    )


@gluon.jit
def _wu_vector_fwd_kernel(
    tinv,
    beta_v,
    beta_kd,
    u,
    w,
    cu_seqlens,
    chunk_indices,
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BT: gl.constexpr,
    BO: gl.constexpr,
):
    """Apply the inverse WY transform independently to every chunk.

    Computes ``u = Tinv @ (beta * V)`` and
    ``W = Tinv @ (beta * Kn * exp(bg))``. Both products are chunk-local and
    therefore run in parallel across chunks and heads.
    """
    chunk = gl.program_id(0)
    head = gl.program_id(1)
    out_block = gl.program_id(2)
    sequence = gl.load(chunk_indices + chunk * 2).to(gl.int32)
    local_chunk = gl.load(chunk_indices + chunk * 2 + 1).to(gl.int32)
    begin = gl.load(cu_seqlens + sequence).to(gl.int32)
    end = gl.load(cu_seqlens + sequence + 1).to(gl.int32)
    length = end - begin
    token0 = local_chunk * BT

    load_t_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    load_x_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    a_layout: gl.constexpr = gl.DotOperandLayout(0, mfma_layout, k_width=8)
    b_layout: gl.constexpr = gl.DotOperandLayout(1, mfma_layout, k_width=8)

    rows_t = gl.arange(0, BT, layout=gl.SliceLayout(1, load_t_layout))
    cols_t = gl.arange(0, BT, layout=gl.SliceLayout(0, load_t_layout))
    t_offsets = ((begin + token0 + rows_t[:, None]) * H + head) * BT + cols_t[None, :]
    t = gl.load(
        tinv + t_offsets,
        mask=(token0 + rows_t[:, None] < length),
        other=0.0,
    )
    lhs = gl.convert_layout(t.to(gl.bfloat16), a_layout)

    rows_x = gl.arange(0, BT, layout=gl.SliceLayout(1, load_x_layout))
    cols_x = gl.arange(0, BO, layout=gl.SliceLayout(0, load_x_layout))
    token_offsets = begin + token0 + rows_x[:, None]
    value_offsets = out_block * BO + cols_x[None, :]
    x_mask = (token0 + rows_x[:, None] < length) & (value_offsets < V)
    bv_offsets = (token_offsets * H + head) * V + value_offsets
    bv = gl.load(beta_v + bv_offsets, mask=x_mask, other=0.0)
    rhs_v = gl.convert_layout(bv.to(gl.bfloat16), b_layout)
    acc_v = gl.zeros([BT, BO], gl.float32, mfma_layout)
    acc_v = cdna4.mfma(lhs, rhs_v, acc_v)

    key_mask = (token0 + rows_x[:, None] < length) & (value_offsets < K)
    bk_offsets = (token_offsets * H + head) * K + value_offsets
    bk = gl.load(beta_kd + bk_offsets, mask=key_mask, other=0.0)
    rhs_k = gl.convert_layout(bk.to(gl.bfloat16), b_layout)
    acc_k = gl.zeros([BT, BO], gl.float32, mfma_layout)
    acc_k = cdna4.mfma(lhs, rhs_k, acc_k)

    out_rows = gl.arange(0, BT, layout=gl.SliceLayout(1, mfma_layout))
    out_cols = gl.arange(0, BO, layout=gl.SliceLayout(0, mfma_layout))
    out_tokens = begin + token0 + out_rows[:, None]
    out_values = out_block * BO + out_cols[None, :]
    gl.store(
        u + (out_tokens * H + head) * V + out_values,
        acc_v.to(gl.bfloat16),
        mask=(token0 + out_rows[:, None] < length) & (out_values < V),
    )
    gl.store(
        w + (out_tokens * H + head) * K + out_values,
        acc_k.to(gl.bfloat16),
        mask=(token0 + out_rows[:, None] < length) & (out_values < K),
    )


@gluon.jit
def _state_scan_fwd_kernel(
    w,
    u,
    kn,
    bg,
    initial_state,
    state_checkpoints,
    vnew,
    final_state,
    cu_seqlens,
    chunk_offsets,
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BT: gl.constexpr,
    BO: gl.constexpr,
):
    """Carry the KDA state sequentially across chunks.

    For every chunk, stores its input-state checkpoint and computes

    ``v_new = u - W @ H``

    ``H = exp(bg_last) * H + (Kn * exp(bg_last - bg))^T @ v_new``.

    One program owns a ``[K, BO]`` tile for a sequence and head, keeping the
    canonical K-major state resident in FP32 across the chunk loop. Packed
    sequences restart from their own initial state and never carry state across
    a sequence boundary. Checkpoints and ``v_new`` allow the heavier output
    matmuls to execute in parallel in a separate kernel.
    """
    value_block = gl.program_id(0)
    sequence_head = gl.program_id(1)
    sequence = sequence_head // H
    head = sequence_head % H
    begin = gl.load(cu_seqlens + sequence).to(gl.int32)
    end = gl.load(cu_seqlens + sequence + 1).to(gl.int32)
    length = end - begin
    num_chunks = gl.cdiv(length, BT)
    chunk_base = gl.load(chunk_offsets + sequence).to(gl.int32)

    uv_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    state_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    uv_a_layout: gl.constexpr = gl.DotOperandLayout(0, uv_layout, k_width=8)
    uv_b_layout: gl.constexpr = gl.DotOperandLayout(1, uv_layout, k_width=8)
    state_a_layout: gl.constexpr = gl.DotOperandLayout(0, state_layout, k_width=8)
    state_b_layout: gl.constexpr = gl.DotOperandLayout(1, state_layout, k_width=8)
    load_w_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])

    state_keys = gl.arange(0, K, layout=gl.SliceLayout(1, state_layout))
    state_values = gl.arange(0, BO, layout=gl.SliceLayout(0, state_layout))
    values = value_block * BO + state_values
    state_offsets = state_keys[:, None] * V + values[None, :]
    state_mask = (state_keys[:, None] < K) & (values[None, :] < V)
    state_base = sequence_head * K * V
    state = gl.load(
        initial_state + state_base + state_offsets,
        mask=state_mask,
        other=0.0,
    ).to(gl.float32)

    w_rows = gl.arange(0, BT, layout=gl.SliceLayout(1, load_w_layout))
    w_keys = gl.arange(0, K, layout=gl.SliceLayout(0, load_w_layout))
    uv_rows = gl.arange(0, BT, layout=gl.SliceLayout(1, uv_layout))
    uv_values = gl.arange(0, BO, layout=gl.SliceLayout(0, uv_layout))
    out_values = value_block * BO + uv_values

    for local_chunk in range(num_chunks):
        checkpoint_base = ((chunk_base + local_chunk).to(gl.int64) * H + head) * K * V
        gl.store(
            state_checkpoints + checkpoint_base + state_offsets,
            state.to(gl.bfloat16),
            mask=state_mask,
        )

        token0 = local_chunk * BT
        token_offsets = begin + token0 + w_rows[:, None]
        w_offsets = (token_offsets * H + head) * K + w_keys[None, :]
        row_mask = token0 + w_rows[:, None] < length
        w_value = gl.load(
            w + w_offsets,
            mask=row_mask & (w_keys[None, :] < K),
            other=0.0,
        )
        lhs = gl.convert_layout(w_value.to(gl.bfloat16), uv_a_layout)
        rhs = gl.convert_layout(state.to(gl.bfloat16), uv_b_layout)
        prediction = gl.zeros([BT, BO], gl.float32, uv_layout)
        prediction = cdna4.mfma(lhs, rhs, prediction)

        uv_tokens = begin + token0 + uv_rows[:, None]
        u_offsets = (uv_tokens * H + head) * V + out_values[None, :]
        uv_mask = (token0 + uv_rows[:, None] < length) & (out_values[None, :] < V)
        u_value = gl.load(u + u_offsets, mask=uv_mask, other=0.0).to(gl.float32)
        new_value = u_value - prediction
        gl.store(
            vnew + u_offsets,
            new_value.to(gl.bfloat16),
            mask=uv_mask,
        )

        bg_offsets = ((begin + token0 + w_rows[:, None]) * H + head) * K + w_keys[
            None, :
        ]
        bg_value = gl.load(
            bg + bg_offsets,
            mask=row_mask & (w_keys[None, :] < K),
            other=0.0,
        ).to(gl.float32)
        key_value = gl.load(
            kn + bg_offsets,
            mask=row_mask & (w_keys[None, :] < K),
            other=0.0,
        ).to(gl.float32)
        valid_rows = token0 + w_rows < length
        bg_last = gl.min(gl.where(valid_rows[:, None], bg_value, float("inf")), axis=0)
        kend = key_value * gl.exp(bg_last[None, :] - bg_value)
        kend = gl.where(valid_rows[:, None], kend, 0.0)
        state_lhs = gl.convert_layout(kend.trans(1, 0).to(gl.bfloat16), state_a_layout)
        state_rhs = gl.convert_layout(new_value.to(gl.bfloat16), state_b_layout)
        state_decay = gl.convert_layout(
            gl.exp(bg_last), gl.SliceLayout(1, state_layout)
        )
        state *= state_decay[:, None]
        state = cdna4.mfma(state_lhs, state_rhs, state)

    gl.store(
        final_state + state_base + state_offsets,
        state,
        mask=state_mask,
    )


@gluon.jit
def _output_fwd_kernel(
    qn,
    kn,
    bg,
    state_checkpoints,
    vnew,
    output,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    SCALE: gl.constexpr,
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BT: gl.constexpr,
    BC: gl.constexpr,
):
    """Compute the fully-parallel output for each chunk.

    Computes ``o = Qd @ H + tril(Qd @ Ki^T, 0) @ v_new``, where
    ``Qd = scale * Qn * exp(bg)`` and ``Ki = Kn * exp(-bg)``. The causal
    intra-chunk term uses the same subchunk reference as the KKt kernel to bound
    ``exp(-bg)``; the inter-chunk term reads the state checkpoint produced by
    the sequential scan.
    """
    chunk = gl.program_id(0)
    head = gl.program_id(1)
    sequence = gl.load(chunk_indices + chunk * 2).to(gl.int32)
    local_chunk = gl.load(chunk_indices + chunk * 2 + 1).to(gl.int32)
    begin = gl.load(cu_seqlens + sequence).to(gl.int32)
    end = gl.load(cu_seqlens + sequence + 1).to(gl.int32)
    length = end - begin
    chunk_base = gl.load(chunk_offsets + sequence).to(gl.int32)

    load_q_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [1, 2], [1, 0])
    qk_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 2],
    )
    out_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, 2],
    )
    pv_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[1, 2],
    )
    qk_a_layout: gl.constexpr = gl.DotOperandLayout(0, qk_layout, k_width=8)
    qk_b_layout: gl.constexpr = gl.DotOperandLayout(1, qk_layout, k_width=8)
    out_a_layout: gl.constexpr = gl.DotOperandLayout(0, out_layout, k_width=8)
    out_b_layout: gl.constexpr = gl.DotOperandLayout(1, out_layout, k_width=8)
    pv_a_layout: gl.constexpr = gl.DotOperandLayout(0, pv_layout, k_width=8)
    pv_b_layout: gl.constexpr = gl.DotOperandLayout(1, pv_layout, k_width=8)

    rows = gl.arange(0, BC, layout=gl.SliceLayout(1, load_q_layout))
    keys = gl.arange(0, K, layout=gl.SliceLayout(0, load_q_layout))
    base = (begin * H + head) * K
    state_values = gl.arange(0, V, layout=gl.SliceLayout(1, load_q_layout))
    state_keys = gl.arange(0, K, layout=gl.SliceLayout(0, load_q_layout))
    checkpoint_base = ((chunk_base + local_chunk).to(gl.int64) * H + head) * K * V
    checkpoint_offsets = state_keys[None, :] * V + state_values[:, None]
    checkpoint = gl.load(
        state_checkpoints + checkpoint_base + checkpoint_offsets,
        mask=(state_keys[None, :] < K) & (state_values[:, None] < V),
        other=0.0,
    ).to(gl.float32)
    checkpoint_dot = gl.convert_layout(
        checkpoint.trans(1, 0).to(gl.bfloat16), out_b_layout
    )
    score_rows = gl.arange(0, BC, layout=gl.SliceLayout(1, qk_layout))
    score_cols = gl.arange(0, BC, layout=gl.SliceLayout(0, qk_layout))
    v_rows = gl.arange(0, BC, layout=gl.SliceLayout(0, load_q_layout))
    out_rows = gl.arange(0, BC, layout=gl.SliceLayout(1, out_layout))
    out_values = gl.arange(0, V, layout=gl.SliceLayout(0, out_layout))

    for row_block in range(BT // BC):
        row0 = local_chunk * BT + row_block * BC
        row_tokens = row0 + rows
        row_mask = row_tokens < length
        reference = gl.load(
            bg + base + row0 * H * K + keys,
            mask=(row0 < length) & (keys < K),
            other=0.0,
        ).to(gl.float32)
        row_offsets = base + row_tokens[:, None] * H * K + keys[None, :]
        q_value = gl.load(
            qn + row_offsets,
            mask=row_mask[:, None] & (keys[None, :] < K),
            other=0.0,
        ).to(gl.float32)
        row_bg = gl.load(
            bg + row_offsets,
            mask=row_mask[:, None] & (keys[None, :] < K),
            other=0.0,
        ).to(gl.float32)

        qd = q_value * gl.exp(row_bg) * SCALE
        qd_dot = gl.convert_layout(qd.to(gl.bfloat16), out_a_layout)
        acc = gl.zeros([BC, V], gl.float32, out_layout)
        acc = cdna4.mfma(qd_dot, checkpoint_dot, acc)

        qref = q_value * gl.exp(row_bg - reference[None, :]) * SCALE
        qref_dot = gl.convert_layout(qref.to(gl.bfloat16), qk_a_layout)
        for col_block in range(row_block + 1):
            col0 = local_chunk * BT + col_block * BC
            col_tokens = col0 + rows
            col_mask = col_tokens < length
            col_offsets = base + col_tokens[:, None] * H * K + keys[None, :]
            k_value = gl.load(
                kn + col_offsets,
                mask=col_mask[:, None] & (keys[None, :] < K),
                other=0.0,
            ).to(gl.float32)
            col_bg = gl.load(
                bg + col_offsets,
                mask=col_mask[:, None] & (keys[None, :] < K),
                other=0.0,
            ).to(gl.float32)
            kref = k_value * gl.exp(reference[None, :] - col_bg)
            kref_dot = gl.convert_layout(kref.trans(1, 0).to(gl.bfloat16), qk_b_layout)
            scores = gl.zeros([BC, BC], gl.float32, qk_layout)
            scores = cdna4.mfma(qref_dot, kref_dot, scores)
            if col_block == row_block:
                scores = gl.where(
                    score_rows[:, None] >= score_cols[None, :], scores, 0.0
                )
            scores_dot = gl.convert_layout(scores.to(gl.bfloat16), pv_a_layout)

            v_offsets = (
                (begin + col0 + v_rows[None, :]) * H + head
            ) * V + state_values[:, None]
            v_block = gl.load(
                vnew + v_offsets,
                mask=(col0 + v_rows[None, :] < length) & (state_values[:, None] < V),
                other=0.0,
            ).to(gl.float32)
            v_dot = gl.convert_layout(v_block.trans(1, 0).to(gl.bfloat16), pv_b_layout)
            intra = gl.zeros([BC, V], gl.float32, pv_layout)
            intra = cdna4.mfma(scores_dot, v_dot, intra)
            acc += gl.convert_layout(intra, out_layout)

        out_offsets = ((begin + row0 + out_rows[:, None]) * H + head) * V + out_values[
            None, :
        ]
        gl.store(
            output + out_offsets,
            acc.to(output.dtype.element_ty),
            mask=(row0 + out_rows[:, None] < length) & (out_values[None, :] < V),
        )


def kda_chunk_prefill_gfx950(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run chunk-parallel KDA prefill on packed GFX950 inputs.

    Args:
        q: Packed queries with shape ``[1,T,H,K]``.
        k: Packed keys with shape ``[1,T,H,K]``.
        v: Packed values with shape ``[1,T,H,V]``.
        g_raw: Raw per-key-channel gates with shape ``[1,T,H,K]``.
        beta_logits: Raw delta coefficients with shape ``[1,T,H]``.
        A_log: Per-head log decay with shape ``[H]``.
        dt_bias: Per-head, per-key-channel gate bias with shape ``[H,K]``.
        initial_state: Initial canonical K-major state ``[N,H,K,V]``.
        cu_seqlens: Packed-sequence prefix sums with shape ``[N+1]``.
        lower_bound: Optional lower bound used by the safe decay gate.

    Returns:
        The packed output ``[1,T,H,V]`` and final state ``[N,H,K,V]``.
    """
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("gfx950 Gluon KDA prefill requires packed [1,T,H,K] inputs")
    if q.shape != k.shape or q.shape != g_raw.shape:
        raise ValueError("q, k, and raw_g must have identical shapes")
    if initial_state.ndim != 4:
        raise ValueError("initial_state must use the canonical [N,H,K,V] layout")

    q = q[0].contiguous()
    k = k[0].contiguous()
    v = v[0].contiguous()
    g_raw = g_raw[0].contiguous()
    beta_logits = beta_logits[0].contiguous()
    heads, key_dim = q.shape[1:]
    value_dim = v.shape[-1]
    if initial_state.shape[1:] != (heads, key_dim, value_dim):
        raise ValueError("initial_state must have shape [N,H,K,V]")
    if key_dim != 128 or value_dim != 128:
        raise ValueError("gfx950 Gluon KDA prefill currently specializes K=V=128")
    cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
    if cu_seqlens.numel() - 1 != initial_state.shape[0]:
        raise ValueError("cu_seqlens and initial_state must describe the same batch")

    from tokenspeed_kernel.ops.attention.triton.linear.cumsum import (
        chunk_local_cumsum_vector,
    )
    from tokenspeed_kernel.ops.attention.triton.linear.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    from tokenspeed_kernel.ops.attention.triton.linear.l2norm import l2norm_fwd
    from tokenspeed_kernel.ops.attention.triton.linear.solve_tril import solve_tril

    chunk_size = _CHUNK_SIZE
    subchunk_size = _SUBCHUNK_SIZE
    total_tokens = q.shape[0]
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)
    num_chunks = chunk_indices.shape[0]
    num_sequences = cu_seqlens.numel() - 1

    gate = torch.empty_like(g_raw, dtype=torch.float32)
    beta = torch.empty_like(beta_logits, dtype=torch.float32)
    block_dim = triton.next_power_of_2(key_dim)
    _kda_prepare_gate_beta_kernel[(total_tokens, heads)](
        g_raw,
        beta_logits,
        A_log.contiguous(),
        dt_bias.view(heads, key_dim).contiguous(),
        gate,
        beta,
        H=heads,
        D=key_dim,
        BLOCK_D=block_dim,
        HAS_LOWER_BOUND=lower_bound is not None,
        LOWER_BOUND=0.0 if lower_bound is None else lower_bound,
        num_warps=min(max(block_dim // 32, 1), 8),
    )
    qn = l2norm_fwd(q, output_dtype=q.dtype)
    kn = l2norm_fwd(k, output_dtype=k.dtype)
    bg = chunk_local_cumsum_vector(
        gate.unsqueeze(0),
        chunk_size,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
    )[0].contiguous()

    beta_kd = torch.empty_like(kn)
    beta_v = torch.empty_like(v, dtype=torch.bfloat16)
    kn_scan = torch.empty_like(k, dtype=torch.bfloat16)
    block_key = triton.next_power_of_2(key_dim)
    block_value = triton.next_power_of_2(value_dim)
    _gate_apply_kernel[(total_tokens * heads,)](
        kn,
        bg,
        beta,
        v,
        beta_kd,
        beta_v,
        kn_scan,
        H=heads,
        K=key_dim,
        V=value_dim,
        BLOCK_K=block_key,
        BLOCK_V=block_value,
        num_warps=min(max(max(block_key, block_value) // 32, 1), 8),
    )

    wy = torch.zeros(
        1,
        total_tokens,
        heads,
        chunk_size,
        device=q.device,
        dtype=torch.float32,
    )
    subchunks = chunk_size // subchunk_size
    _kkt_vector_fwd_kernel[(num_chunks * subchunks, heads)](
        kn,
        bg,
        beta,
        wy,
        cu_seqlens,
        chunk_indices,
        H=heads,
        K=key_dim,
        BT=chunk_size,
        BC=subchunk_size,
        num_warps=1,
    )
    tinv = solve_tril(
        A=wy,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.bfloat16,
    )

    u = torch.empty_like(v, dtype=torch.bfloat16)
    w = torch.empty_like(k, dtype=torch.bfloat16)
    output_block = 128
    _wu_vector_fwd_kernel[
        (num_chunks, heads, triton.cdiv(max(key_dim, value_dim), output_block))
    ](
        tinv,
        beta_v,
        beta_kd,
        u,
        w,
        cu_seqlens,
        chunk_indices,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BO=output_block,
        num_warps=4,
    )

    state_checkpoints = torch.empty(
        num_chunks,
        heads,
        key_dim,
        value_dim,
        device=q.device,
        dtype=torch.bfloat16,
    )
    vnew = torch.empty_like(v, dtype=torch.bfloat16)
    final_state = torch.empty_like(initial_state)
    scan_output_block = 16
    _state_scan_fwd_kernel[
        (triton.cdiv(value_dim, scan_output_block), num_sequences * heads)
    ](
        w,
        u,
        kn_scan,
        bg,
        initial_state.contiguous(),
        state_checkpoints,
        vnew,
        final_state,
        cu_seqlens,
        chunk_offsets,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BO=scan_output_block,
        num_warps=4,
    )

    output = torch.empty_like(v)
    _output_fwd_kernel[(num_chunks, heads)](
        qn,
        kn,
        bg,
        state_checkpoints,
        vnew,
        output,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        key_dim**-0.5,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BC=subchunk_size,
        num_warps=2,
    )
    return output.unsqueeze(0), final_state
