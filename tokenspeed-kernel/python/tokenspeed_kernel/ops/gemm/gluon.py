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

"""Registration shim for AMD Gluon GEMM kernels.

Dense16 Gluon GEMM registration is intentionally disabled so dense GEMM
dispatch falls back to the portable torch GEMM implementation.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if current_platform().is_amd:
    from tokenspeed_kernel_amd.ops.gemm.gated_rmsnorm_linear_gfx950 import (
        gluon_gated_rmsnorm_linear_gfx950 as _gated_rmsnorm_linear_impl,
    )
    from tokenspeed_kernel_amd.ops.gemm.linear_attnres_partials_gfx950 import (
        gluon_linear_attnres_partials_gfx950 as _linear_attnres_partials_impl,
    )
    from tokenspeed_kernel_amd.ops.gemm.moe_input_projections_gfx950 import (
        gluon_moe_input_projections_gfx950 as _moe_input_projections_impl,
    )
    from tokenspeed_kernel_amd.ops.gemm.rmsnorm_linear_add_gfx950 import (
        gluon_rmsnorm_linear_add_gfx950 as _rmsnorm_linear_add_impl,
    )

    @register_kernel(
        "gemm",
        "gated_rmsnorm_linear",
        name="gluon_gated_rmsnorm_linear_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    x=dense_tensor_format(torch.bfloat16),
                    gate=dense_tensor_format(torch.bfloat16),
                    norm_weight=dense_tensor_format(torch.bfloat16),
                    linear_weight=dense_tensor_format(torch.bfloat16),
                    out=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "tokens": frozenset({1}),
            "group_size": frozenset({64, 128, 256}),
            "num_groups": frozenset(range(1, 33)),
            "output_size": frozenset({4096, 5120, 6144, 7168, 8192}),
            "gate_kind": frozenset({"sigmoid", "silu"}),
            "inputs_contiguous": frozenset({True}),
        },
    )
    def gluon_gated_rmsnorm_linear_gfx950(*args, **kwargs):
        return _gated_rmsnorm_linear_impl(*args, **kwargs)

    @register_kernel(
        "gemm",
        "linear_attnres_partials",
        name="gluon_linear_attnres_partials_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    hidden_states=dense_tensor_format(torch.bfloat16),
                    weight=dense_tensor_format(torch.bfloat16),
                    blocks=dense_tensor_format(torch.bfloat16),
                    score_weight_a=dense_tensor_format(torch.bfloat16),
                    score_weight_b=dense_tensor_format(torch.bfloat16),
                    out=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "tokens": frozenset({1}),
            "input_size": frozenset({7168}),
            "output_size": frozenset({3648, 6288}),
            "num_blocks": frozenset(range(1, 12)),
            "inputs_contiguous": frozenset({True}),
        },
    )
    def gluon_linear_attnres_partials_gfx950(**kwargs):
        return _linear_attnres_partials_impl(**kwargs)

    @register_kernel(
        "gemm",
        "moe_input_projections",
        name="gluon_moe_input_projections_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    hidden_states=dense_tensor_format(torch.bfloat16),
                    router_weight=dense_tensor_format(torch.bfloat16),
                    routed_weight=dense_tensor_format(torch.bfloat16),
                    shared_gate_up_weight=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "tokens": frozenset({1}),
            "hidden_size": frozenset({7168}),
            "num_experts": frozenset({896}),
            "latent_size": frozenset({3584}),
            "shared_size": frozenset({768}),
            "inputs_contiguous": frozenset({True}),
        },
    )
    def gluon_moe_input_projections_gfx950(**kwargs):
        return _moe_input_projections_impl(
            kwargs["hidden_states"],
            kwargs["router_weight"],
            kwargs["routed_weight"],
            kwargs["shared_gate_up_weight"],
            beta=kwargs["gate_clamp"],
            linear_beta=kwargs["up_clamp"],
        )

    @register_kernel(
        "gemm",
        "rmsnorm_linear_add",
        name="gluon_rmsnorm_linear_add_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    hidden_states=dense_tensor_format(torch.bfloat16),
                    norm_weight=dense_tensor_format(torch.bfloat16),
                    linear_weight=dense_tensor_format(torch.bfloat16),
                    out=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        priority=Priority.SPECIALIZED,
        traits={
            "tokens": frozenset({1}),
            "input_size": frozenset({3584}),
            "output_size": frozenset({7168}),
            "num_addends": frozenset({2}),
            "inputs_contiguous": frozenset({True}),
        },
    )
    def gluon_rmsnorm_linear_add_gfx950(**kwargs):
        addends = kwargs.pop("addends")
        return _rmsnorm_linear_add_impl(
            kwargs.pop("hidden_states"),
            kwargs.pop("norm_weight"),
            kwargs.pop("linear_weight"),
            *addends,
            **kwargs,
        )
