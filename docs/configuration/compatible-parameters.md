# Compatible Parameters

TokenSpeed keeps familiar serving parameter names when the operational meaning
is the same. This makes recipes portable while still documenting
TokenSpeed-specific behavior explicitly.

## Directly Aligned

| Parameter | TokenSpeed behavior |
| --- | --- |
| positional `model` | Model path or Hugging Face repo ID. |
| `--model` | Equivalent to positional `model`. |
| `--tokenizer` | Tokenizer path. |
| `--tokenizer-mode` | Tokenizer implementation mode. |
| `--skip-tokenizer-init` | Skip tokenizer initialization. |
| `--load-format` | Weight loading format. |
| `--trust-remote-code` | Allow custom model code from the model repository. |
| `--dtype` | Weight and activation dtype. |
| `--kv-cache-dtype` | KV cache storage dtype. |
| `--quantization` | Weight quantization method. |
| `--quantization-param-path` | KV cache scaling-factor file. |
| `--max-model-len` | Maximum sequence length. |
| `--device` | Device type. TokenSpeed currently serves CUDA. |
| `--served-model-name` | OpenAI-compatible served model name. |
| `--revision` | Model revision. |
| `--download-dir` | Model download directory. |
| `--hf-overrides` | JSON model config overrides. |
| `--host` | HTTP bind host. |
| `--port` | HTTP bind port. |
| `--api-key` | API key for the server. |
| `--chat-template` | Chat template name or path. |
| `--gpu-memory-utilization` | GPU memory fraction used for weights and KV cache. |
| `--max-num-seqs` | Maximum concurrent sequences. |
| `--block-size` | KV cache block size. |
| `--enable-prefix-caching` | Enable prefix cache reuse. |
| `--no-enable-prefix-caching` | Disable prefix cache reuse. |
| `--enforce-eager` | Disable CUDA graph execution. |
| `--max-cudagraph-capture-size` | Largest CUDA graph capture size. |
| `--tensor-parallel-size`, `--tp` | Set attention tensor parallel size. |
| `--data-parallel-size` | Data parallel size. |
| `--mm-encoder-tp-mode` | Select multimodal encoder weight TP (`weights`) or item data parallelism (`data`). |
| `--enable-expert-parallel` | Enable expert parallelism. |
| `--speculative-config` | JSON speculative decoding config. |
| `--kv-events-config` | JSON KV cache event publisher config; the vLLM-style `enable_kv_cache_events` field is accepted and defaults to ZMQ when enabled. |
| `--tool-call-parser` | OpenAI-compatible tool-call parser. |
| `--reasoning-parser` | Reasoning-output parser. |

## Similar But Not Identical

| Recipe parameter | TokenSpeed parameter | Difference |
| --- | --- | --- |
| `--max-num-batched-tokens` | `--chunked-prefill-size` | TokenSpeed uses this as the scheduler per-iteration issue budget. |
| `--max-num-batched-tokens` | `--max-total-tokens` | TokenSpeed uses this for the global token pool size override. |
| `--tensor-parallel-size`, `--tp` | `--attn-tp-size` | The familiar alias maps to attention TP. TokenSpeed can split attention, dense, and MoE TP. |
| `--expert-parallel-size` | `--expert-parallel-size`, `--ep-size` | TokenSpeed supports the familiar name and its existing short form. |
| `--attention-backend` | `--attention-backend` | Name is aligned; available backend values are TokenSpeed-specific. |
| `--moe-backend` | `--moe-backend` | Name is aligned; available backend values are TokenSpeed-specific. |

## Recipe Translation Notes

- Use `tokenspeed serve` as the launcher.
- Pass the model path positionally, then keep `--trust-remote-code`, `--max-model-len`, `--kv-cache-dtype`, `--gpu-memory-utilization`, `--max-num-seqs`, `--tensor-parallel-size`, `--reasoning-parser`, and `--tool-call-parser` when the model needs them.
- Review `--max-num-batched-tokens` before copying it. TokenSpeed usually wants `--chunked-prefill-size` for per-iteration scheduling.
- Review backend names. TokenSpeed backends are optimized for its runtime and kernel packages.
- Keep TokenSpeed-specific `--attn-tp-size`, `--moe-tp-size`, `--disaggregation-*`, and `--kvstore-*` only when the deployment needs those features.

## Seeded Sampling

OpenAI-compatible generation requests may provide `seed`. The TokenSpeed SMG
adapter preserves a seed carried in the protobuf extension fields. The SMG
version currently pinned by TokenSpeed does not yet serialize the OpenAI seed;
when that field is absent, the adapter falls back to a configured server
`--seed`. An explicit extension-field seed takes precedence, and deployments
that do not configure `--seed` retain request-ID-derived sampling.

The Triton sampling backend advances the seed by the number of output tokens
committed for the request. Prompt length, prefix-cache hits, request-pool
placement, and decode batch shape therefore do not change the random draw
sequence for a fixed seed.

When every request in a decode batch has an explicit seed, graph replay keeps
using the largest configured capture bucket as requests finish. The kernels and
collective tensor shapes therefore remain fixed for the seeded batch. Unseeded
traffic continues to use the smallest fitting captured bucket.

The scheduler holds a newly arriving seeded request burst for a 250 ms quiet
window (capped at 1 second), then orders it by tokenized model input. The C++
scheduler retains that immutable input ordering across prefill and decode,
instead of using frontend-generated request IDs as its primary tie-breaker.
This prevents gateway timing and random IDs from changing the prefill
partition or permuting packed rows. Unseeded traffic retains the nonblocking
admission path.

`/flush_cache` invalidates the scheduler-owned FlatKV and radix/hybrid prefix
caches before acknowledging success. The scheduler must be drained; a flush
requested while model or cache-transfer work is active is rejected rather than
reporting a false success. This ensures cache-flushed evaluation repeats start
from the same prefill state.

On gfx950, grouped K3 A16W4 SiTU EP8 route outputs are combined through
per-route BF16 partials and a fixed-order FP32 reduction instead of atomic
accumulation.

The SMG health RPC uses a scheduler load round-trip rather than generating a
token. Health monitoring therefore cannot enter the inference batch, change
its graph padding, or perturb the logits of live requests.
