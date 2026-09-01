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
"""Standalone benchmark for Kimi-K3's full-attention (NoPE MLA) kernels.

This benchmarks one tensor-parallel rank of the attention kernel itself. It
does not include projections, the output gate, collectives, KDA layers, or the
scheduler. The constants match Kimi-K3 at TP=8 and the production FP8 KV-cache
path used by TokenSpeed on MI355X.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from tokenspeed_kernel import mla_decode_with_kvcache, mla_prefill
from tokenspeed_kernel.registry import load_builtin_kernels

NUM_ATTENTION_LAYERS = 24
NUM_LOCAL_HEADS = 12
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM = 128
KV_LORA_RANK = 512
ABSORBED_QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
PAGE_SIZE = 64
SOFTMAX_SCALE = QK_HEAD_DIM**-0.5
ATTENTION_DTYPE = torch.float8_e4m3fn
OUTPUT_DTYPE = torch.bfloat16


@dataclass(frozen=True)
class Case:
    """A single Kimi-K3 MLA kernel workload."""

    mode: str
    batch_size: int
    query_length: int
    context_length: int
    prompt_length: int = 0
    speculative_tokens: int = 1
    replay_chunks: int = 1


@dataclass(frozen=True)
class Result:
    """Timing result for one workload."""

    mode: str
    batch_size: int
    query_length: int
    context_length: int
    prompt_length: int
    speculative_tokens: int
    replay_chunks: int
    median_us: float
    p90_us: float
    per_model_forward_ms: float


def parse_positive_ints(value: str) -> list[int]:
    """Parse a comma-separated list of positive integers."""
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def make_cases(
    mode: str,
    prefill_lengths: list[int],
    decode_contexts: list[int],
    decode_batches: list[int],
    speculative_tokens: int,
) -> list[Case]:
    """Build the requested benchmark matrix.

    A 50K or 131K prompt is processed in 8K chunks. ``prefill_replay`` models
    one non-causal attention call from the final prompt chunk to a prior 8K
    chunk; ``replay_chunks`` says how many such calls the final chunk requires.
    """
    cases: list[Case] = []
    if mode in ("all", "prefill"):
        for length in prefill_lengths:
            chunk_length = length % 8192 or min(length, 8192)
            cases.append(
                Case(
                    mode="prefill_causal",
                    batch_size=1,
                    query_length=chunk_length,
                    context_length=chunk_length,
                    prompt_length=length,
                )
            )
            prior_chunks = max(math.ceil(length / 8192) - 1, 0)
            if prior_chunks:
                cases.append(
                    Case(
                        mode="prefill_replay",
                        batch_size=1,
                        query_length=chunk_length,
                        context_length=8192,
                        prompt_length=length,
                        replay_chunks=prior_chunks,
                    )
                )
    if mode in ("all", "decode"):
        cases.extend(
            Case(
                mode="decode",
                batch_size=batch_size,
                query_length=1,
                context_length=context_length,
                speculative_tokens=speculative_tokens,
            )
            for context_length in decode_contexts
            for batch_size in decode_batches
        )
    return cases


def _make_fp8(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return (torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.25).to(
        ATTENTION_DTYPE
    )


def _make_prefill_call(
    case: Case,
    device: torch.device,
    solution: str | None,
    kernel: str | None,
) -> Callable[[], object]:
    q = _make_fp8((case.query_length, NUM_LOCAL_HEADS, QK_HEAD_DIM), device)
    k = _make_fp8((case.context_length, NUM_LOCAL_HEADS, QK_HEAD_DIM), device)
    v = _make_fp8((case.context_length, NUM_LOCAL_HEADS, V_HEAD_DIM), device)
    cu_q = torch.tensor([0, case.query_length], dtype=torch.int32, device=device)
    cu_kv = torch.tensor([0, case.context_length], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([case.context_length], dtype=torch.int32, device=device)
    out = torch.empty(
        (case.query_length, NUM_LOCAL_HEADS, V_HEAD_DIM),
        dtype=OUTPUT_DTYPE,
        device=device,
    )
    is_causal = case.mode == "prefill_causal"

    def run() -> object:
        return mla_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            max_seqlen_q=case.query_length,
            max_seqlen_kv=case.context_length,
            softmax_scale=SOFTMAX_SCALE,
            seq_lens_kv=seq_lens,
            is_causal=is_causal,
            out=out,
            solution=solution,
            override=kernel,
        )

    return run


def _make_decode_call(
    case: Case,
    device: torch.device,
    solution: str | None,
    kernel: str | None,
) -> Callable[[], object]:
    pages_per_request = math.ceil(case.context_length / PAGE_SIZE)
    num_pages = case.batch_size * pages_per_request
    q_rows = case.batch_size * case.speculative_tokens
    q = _make_fp8((q_rows, 1, NUM_LOCAL_HEADS, ABSORBED_QK_HEAD_DIM), device)
    kv_cache = _make_fp8((num_pages, PAGE_SIZE, 1, ABSORBED_QK_HEAD_DIM), device)
    base_table = torch.arange(num_pages, dtype=torch.int32, device=device).view(
        case.batch_size, pages_per_request
    )
    page_table = base_table.repeat_interleave(case.speculative_tokens, dim=0)
    # Target verification exposes consecutive prefix lengths ending at the
    # target sequence length, matching MLAAttnBackend.forward_decode.
    offsets = torch.arange(
        1 - case.speculative_tokens, 1, dtype=torch.int32, device=device
    ).repeat(case.batch_size)
    cache_seqlens = offsets + case.context_length
    out = torch.empty(
        (q_rows, 1, NUM_LOCAL_HEADS, KV_LORA_RANK),
        dtype=OUTPUT_DTYPE,
        device=device,
    )

    def run() -> object:
        return mla_decode_with_kvcache(
            q=q,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=case.context_length,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            softmax_scale=SOFTMAX_SCALE,
            out=out,
            solution=solution,
            override=kernel,
        )

    return run


def _percentile(sorted_values: list[float], fraction: float) -> float:
    index = math.ceil(fraction * len(sorted_values)) - 1
    return sorted_values[max(index, 0)]


def benchmark_case(
    case: Case,
    device: torch.device,
    warmup_iterations: int,
    iterations: int,
    solution: str | None,
    kernel: str | None,
) -> Result:
    """Allocate and time one case using device events."""
    make_call = _make_decode_call if case.mode == "decode" else _make_prefill_call
    run = make_call(case, device, solution, kernel)
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            run()
        torch.cuda.synchronize(device)

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for start, end in zip(starts, ends, strict=True):
            start.record()
            run()
            end.record()
        torch.cuda.synchronize(device)

    times_us = sorted(
        start.elapsed_time(end) * 1000.0
        for start, end in zip(starts, ends, strict=True)
    )
    median_us = statistics.median(times_us)
    model_multiplier = NUM_ATTENTION_LAYERS * case.replay_chunks
    return Result(
        **asdict(case),
        median_us=median_us,
        p90_us=_percentile(times_us, 0.90),
        per_model_forward_ms=median_us * model_multiplier / 1000.0,
    )


def _format_results(results: list[Result]) -> str:
    header = (
        "| mode | batch | prompt | q | context | spec tokens | repeats | "
        "median (us) | p90 (us) | 24-layer estimate (ms) |"
    )
    divider = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    rows = [header, divider]
    for result in results:
        rows.append(
            f"| {result.mode} | {result.batch_size} | {result.prompt_length or '-'} | "
            f"{result.query_length} | {result.context_length} | "
            f"{result.speculative_tokens} | {result.replay_chunks} | "
            f"{result.median_us:.2f} | "
            f"{result.p90_us:.2f} | {result.per_model_forward_ms:.3f} |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Kimi-K3 TP8 NoPE-MLA kernels on one GPU"
    )
    parser.add_argument("--mode", choices=("all", "prefill", "decode"), default="all")
    parser.add_argument(
        "--prefill-lengths",
        type=parse_positive_ints,
        default=[4096, 8192, 50_000, 131_072],
        help="Prompt lengths; long prompts are represented as 8K chunks",
    )
    parser.add_argument(
        "--decode-contexts",
        type=parse_positive_ints,
        default=[4096, 50_000, 131_072],
    )
    parser.add_argument(
        "--decode-batches",
        type=parse_positive_ints,
        default=[1, 2, 4, 8, 16],
    )
    parser.add_argument(
        "--speculative-tokens",
        type=int,
        default=1,
        help="Use 5 to model one target verification of four EAGLE3 drafts",
    )
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--solution", default="gluon")
    parser.add_argument("--kernel", help="Optional exact registered kernel override")
    parser.add_argument("--json", type=Path, help="Optional JSON result path")
    args = parser.parse_args(argv)

    if args.speculative_tokens <= 0:
        parser.error("--speculative-tokens must be positive")
    if args.warmup_iterations < 0 or args.iterations <= 0:
        parser.error("warmup iterations must be non-negative and iterations positive")
    if not torch.cuda.is_available():
        parser.error("a CUDA/HIP device is required")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    load_builtin_kernels()
    cases = make_cases(
        args.mode,
        args.prefill_lengths,
        args.decode_contexts,
        args.decode_batches,
        args.speculative_tokens,
    )
    results: list[Result] = []
    for case in cases:
        result = benchmark_case(
            case,
            device,
            args.warmup_iterations,
            args.iterations,
            args.solution or None,
            args.kernel,
        )
        results.append(result)
        print(_format_results([result]), flush=True)

    print("\nCombined results\n")
    print(_format_results(results))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([asdict(result) for result in results], indent=2)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
