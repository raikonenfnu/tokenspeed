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

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.sampling import argmax as sampling_argmax

from tokenspeed.runtime.sampling.backends.base import (
    SamplingBackend,
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.backends.greedy import _verify_chain_greedy
from tokenspeed.runtime.sampling.registry import register_backend
from tokenspeed.runtime.sampling.utils import gather_token_logprobs_torch
from tokenspeed.runtime.utils.nvtx import nvtx_range
from tokenspeed.runtime.utils.pdl import pdl_enabled

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams


class TorchSamplingBackend(SamplingBackend):
    """Vendor-neutral pure-torch sampling backend.

    Implements temperature / top_k / top_p (nucleus) sampling for the
    single-step path using only ``torch`` ops plus the portable
    ``tokenspeed_kernel`` argmax. Imports *no* flashinfer kernels, so it is
    safe on platforms (e.g. ROCm/gfx950) where the flashinfer sampling
    kernels are unavailable or crash at launch.

    verify() is greedy-only (chain-greedy), matching the greedy backend;
    stochastic speculative verification is not supported. Sampling RNG runs
    via ``torch.multinomial`` and is therefore not CUDA-graph capturable — use
    this backend with ``enforce_eager`` (it is intended as the sampling
    fallback for non-NVIDIA single-/small-batch serving).
    """

    _HAS_POOL_STATE = True
    _SUPPORTS_DP_VERIFY = False

    def __init__(self, config: SamplingBackendConfig) -> None:
        super().__init__(config)

        pool_rows = config.max_req_pool_size + 1
        self._temperature_pool = torch.ones(
            (pool_rows,), dtype=torch.float32, device=config.device
        )
        self._top_k_pool = torch.full(
            (pool_rows,), -1, dtype=torch.int64, device=config.device
        )
        self._top_p_pool = torch.ones(
            (pool_rows,), dtype=torch.float32, device=config.device
        )

        self._ones_buf = torch.ones(
            (config.max_bs,), dtype=torch.int32, device=config.device
        )
        self._predict_buf = torch.zeros(
            (config.max_bs * config.max_draft_tokens_per_req,),
            dtype=torch.int32,
            device=config.device,
        )
        self._accept_index_buf = torch.zeros(
            (config.max_bs * config.max_draft_tokens_per_req,),
            dtype=torch.int32,
            device=config.device,
        )
        self._accept_length_buf = torch.zeros(
            (config.max_bs,), dtype=torch.int32, device=config.device
        )

        self._generator = torch.Generator(device=config.device)
        self._generator.manual_seed(config.random_seed)

    def _reset_slot(self, pool_idx: int, sp: SamplingParams) -> None:
        self._temperature_pool[pool_idx].fill_(float(sp.temperature))
        # top_k <= 0 means "disabled"; store -1 so the filter is skipped.
        top_k = int(sp.top_k)
        self._top_k_pool[pool_idx].fill_(top_k if top_k > 0 else -1)
        self._top_p_pool[pool_idx].fill_(float(sp.top_p))

    @nvtx_range("sampling:sample", color="yellow")
    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = logits_output.next_token_logits
        if sampling_info.vocab_mask is not None:
            sampling_info.apply_vocab_mask(
                logits=logits, vocab_mask=sampling_info.vocab_mask
            )
        bs = logits.shape[0]

        if sampling_info.is_all_greedy:
            tokens = sampling_argmax(logits).to(torch.int32)
        else:
            pool_idx = sampling_info.req_pool_indices.long()
            temperatures = self._temperature_pool[pool_idx].unsqueeze(1)
            top_ks = self._top_k_pool[pool_idx]
            top_ps = self._top_p_pool[pool_idx]

            probs = torch.softmax(
                logits.float() / temperatures.clamp_min(1e-5), dim=-1
            )
            tokens = self._top_k_top_p_sample(probs, top_ks, top_ps).to(torch.int32)

        self.maybe_broadcast(tokens)

        if self.config.enable_output_logprobs:
            logits_output.next_token_logprobs = gather_token_logprobs_torch(
                logits, tokens
            )

        return tokens, self._ones_buf[:bs]

    def _top_k_top_p_sample(
        self,
        probs: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
    ) -> torch.Tensor:
        vocab = probs.shape[-1]
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)

        # top_k: -1 (disabled) keeps the full vocab.
        k = torch.where(top_ks > 0, top_ks, torch.full_like(top_ks, vocab))
        rank = torch.arange(vocab, device=probs.device).unsqueeze(0)
        top_k_keep = rank < k.clamp(max=vocab).unsqueeze(1)

        cumsum = torch.cumsum(sorted_probs, dim=-1)
        # Mass strictly before each token stays under top_p; always retains the
        # top-1 token and the token that crosses the threshold.
        top_p_keep = (cumsum - sorted_probs) < top_ps.unsqueeze(1)

        keep = top_k_keep & top_p_keep
        sorted_probs = torch.where(
            keep, sorted_probs, torch.zeros_like(sorted_probs)
        )
        sorted_probs = sorted_probs / sorted_probs.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-20)

        choice = torch.multinomial(
            sorted_probs, num_samples=1, generator=self._generator
        )
        return sorted_idx.gather(-1, choice).squeeze(-1)

    @nvtx_range("sampling:verify", color="yellow")
    def verify(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bs = candidates.shape[0]
        num_tokens_per_req = candidates.shape[1]

        predict = self._predict_buf[: bs * num_tokens_per_req]
        accept_index = (
            self._accept_index_buf[: bs * num_tokens_per_req]
            .view(bs, num_tokens_per_req)
            .fill_(-1)
        )
        accept_length = self._accept_length_buf[:bs]

        logits = logits_output.next_token_logits
        if sampling_info.vocab_mask is not None:
            sampling_info.apply_vocab_mask(
                logits=logits, vocab_mask=sampling_info.vocab_mask
            )
        target_predict = sampling_argmax(logits).reshape(bs, num_tokens_per_req)

        _verify_chain_greedy(
            predicts=predict,
            accept_index=accept_index,
            accept_token_num=accept_length,
            candidates=candidates.to(torch.int32),
            target_predict=target_predict,
            batch_size=bs,
            num_draft_tokens=num_tokens_per_req,
            enable_pdl=pdl_enabled(),
        )

        accept_length += 1

        if self.config.enable_output_logprobs:
            logits_output.next_token_logprobs = gather_token_logprobs_torch(
                logits, predict
            )

        return predict, accept_length


register_backend("torch", TorchSamplingBackend)
