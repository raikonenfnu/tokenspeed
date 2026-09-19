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

"""Optional AITER FlashKDA prefill specialization for GFX950."""

from __future__ import annotations

import math

import torch

from tokenspeed_kernel.ops.attention.kda import KdaPrefillResult
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.aiter import get_aiter_flash_kda


_flash_kda_fwd = get_aiter_flash_kda()

if current_platform().is_amd and _flash_kda_fwd is not None:

    @register_kernel(
        "attention",
        "kda_paged_prefill",
        name="aiter_flashkda_paged_prefill_gfx950",
        solution="flashkda",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            ("q", "k", "v"),
            "dense",
            {torch.bfloat16},
        ),
        priority=Priority.SPECIALIZED + 1,
        traits={"recurrent_layout": frozenset({"v_major"})},
        tags={"amd", "gfx950", "paged_cache", "fusion"},
    )
    def aiter_flashkda_paged_prefill_gfx950(
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
        cu_seqlens_cpu: torch.Tensor,
        lower_bound: float | None,
    ) -> KdaPrefillResult:
        """Run fused gate preparation and segmented KDA recurrence."""
        del cu_seqlens_cpu
        if lower_bound is None:
            raise ValueError("AITER FlashKDA requires a safe-gate lower bound")
        out, final_state = _flash_kda_fwd(
            q=q,
            k=k,
            v=v,
            g=g_raw,
            beta=beta_logits,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=1.0 / math.sqrt(q.shape[-1]),
            lower_bound=float(lower_bound),
            initial_state=initial_state,
            output_final_state=True,
            state_v_first=True,
            cu_seqlens=cu_seqlens,
        )
        assert final_state is not None
        return KdaPrefillResult(out=out, final_state=final_state)
