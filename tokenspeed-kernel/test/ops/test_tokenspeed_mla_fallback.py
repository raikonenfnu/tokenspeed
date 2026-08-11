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

import torch
from tokenspeed_kernel.ops.attention.tokenspeed_mla import (
    _torch_mla_kv_pack_quantize_fp8,
)


def test_torch_mla_kv_pack_quantize_fp8() -> None:
    k_nope = torch.randn(7, 3, 16, dtype=torch.bfloat16)
    k_pe = torch.randn(7, 1, 8, dtype=torch.bfloat16)
    v = torch.randn(7, 3, 12, dtype=torch.bfloat16)
    k_out = torch.empty(7, 3, 24, dtype=torch.float8_e4m3fn)
    v_out = torch.empty(7, 3, 12, dtype=torch.float8_e4m3fn)

    k, quantized_v = _torch_mla_kv_pack_quantize_fp8(
        k_nope,
        k_pe,
        v,
        k_scale_inv=0.5,
        v_scale_inv=1.5,
        k_out=k_out,
        v_out=v_out,
    )

    expected_k = (
        torch.cat((k_nope, k_pe.expand(-1, 3, -1)), dim=-1)
        .float()
        .mul_(0.5)
        .to(torch.float8_e4m3fn)
    )
    expected_v = v.float().mul_(1.5).to(torch.float8_e4m3fn)
    assert k.data_ptr() == k_out.data_ptr()
    assert quantized_v.data_ptr() == v_out.data_ptr()
    assert torch.equal(k.view(torch.uint8), expected_k.view(torch.uint8))
    assert torch.equal(quantized_v.view(torch.uint8), expected_v.view(torch.uint8))
