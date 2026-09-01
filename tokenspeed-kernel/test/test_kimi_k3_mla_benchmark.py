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

import argparse

import pytest
from tokenspeed_kernel.benchmark.kimi_k3_mla import make_cases, parse_positive_ints


def test_parse_positive_ints() -> None:
    assert parse_positive_ints("1, 4,16") == [1, 4, 16]


@pytest.mark.parametrize("value", ["", "0", "1,-2"])
def test_parse_positive_ints_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_positive_ints(value)


def test_long_prefill_models_causal_chunk_and_prefix_replay() -> None:
    cases = make_cases("prefill", [50_000], [], [], speculative_tokens=1)

    assert [case.mode for case in cases] == ["prefill_causal", "prefill_replay"]
    assert cases[0].query_length == 848
    assert cases[1].context_length == 8192
    assert cases[1].prompt_length == 50_000
    assert cases[1].replay_chunks == 6


def test_decode_matrix_preserves_logical_batch_and_speculation() -> None:
    cases = make_cases("decode", [], [4096, 50_000], [1, 4], speculative_tokens=5)

    assert len(cases) == 4
    assert {(case.batch_size, case.context_length) for case in cases} == {
        (1, 4096),
        (4, 4096),
        (1, 50_000),
        (4, 50_000),
    }
    assert all(case.speculative_tokens == 5 for case in cases)
