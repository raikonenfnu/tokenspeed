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

"""Optional AITER import boundary."""

from __future__ import annotations

from functools import lru_cache
from types import ModuleType
from typing import Callable

from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton


@lru_cache(maxsize=1)
def get_aiter() -> ModuleType | None:
    """Return AITER when its required MLA prefill surface is installed."""
    try:
        # AITER imports its kernels through the canonical ``triton`` package
        # name.  Bind those imports to TokenSpeed's vendored Triton so AITER,
        # Iris, and in-tree kernels share one JITFunction class hierarchy.
        with redirect_triton_to_tokenspeed_triton():
            import aiter
    except ImportError:
        return None

    required = (
        "get_ps_metadata_info_v1",
        "get_ps_metadata_v1",
        "mla_prefill_ps_asm_fwd",
        "mla_reduce_v1",
    )
    if not all(hasattr(aiter, name) for name in required):
        return None
    return aiter


@lru_cache(maxsize=1)
def get_aiter_flash_kda() -> Callable | None:
    """Return AITER's fused KDA prefill entry point when installed."""
    try:
        with redirect_triton_to_tokenspeed_triton():
            from aiter.ops.triton._triton_kernels.chunk_delta_attn.flash_kda import (
                flash_kda_fwd,
            )
    except ImportError:
        return None
    return flash_kda_fwd
