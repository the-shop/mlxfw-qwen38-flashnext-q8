# 2026-10-06: llama.cpp v0.6.0, copy mode, MTP

M5 Max, 128 GiB. Hot cache unless noted. Varied prompts, median of 5 measured runs after 20
warm-ups, 128 tokens out. Greedy outputs were checked byte-identical across configs.
"Peak wired" is system-wide wired memory and includes ~5-6 GiB of macOS baseline.
Between measurement windows the same config moved by about +/-4 tok/s, so compare rows
inside one window, not across windows.

## Copy mode (`LLAMA_GPU_NO_HOST_PTR=1`)

With mmap on, llama.cpp gives the GPU backend one host-pointer buffer per model file that
spans from the first to the last GPU tensor in that file (`src/llama-model.cpp`, the
`buffer_from_host_ptr` path). With `--cpu-moe` the GPU tensors are small and scattered, so
the span covers almost the whole file and macOS wires all of it.

| config, stock-equivalent build | peak wired | result |
|---|---|---|
| `-ngl 99 --cpu-moe` | 69.1 GiB | Metal OOM (HTTP 500) |
| `-ngl 99 --n-cpu-moe 40` | 92.1 GiB | Metal OOM (HTTP 500) |
| `-ngl 99 --cpu-moe`, copy mode | 12.7 GiB | runs, GPU buffer 4.85 GiB |

The patch (`the-shop/llama.cpp`, branch `mlxfw-copy-mode`, one `if`) makes non-CPU devices
allocate their own buffer and copy their tensors. CPU tensors stay mmap'd.

## Results by window

**Window A, services stopped, warm cache**

| config | tok/s | peak wired |
|---|---|---|
| stock #27742 `-ngl 24 -t 12` | 22.09 | 66.9 GiB |
| stock #27742 `-ngl 24 -t 18` | 14.25 | 66.9 GiB |
| v0.6.0 `-ngl 24 -t 12` | 23.63 | 66.5 GiB |
| v0.6.0 `-ngl 24 -t 12` + MTP | 19.77 | 74.0 GiB |

**Window B, background backup running (noisy)**

| config | tok/s | peak wired |
|---|---|---|
| v0.6.0 `-ngl 24` | 16.93 | 67.5 GiB |
| copy `--n-cpu-moe 32` | 22.05 | 52.3 GiB |
| copy `--n-cpu-moe 28` | 21.97 | 62.1 GiB |

**Window C, clean host, copy mode + MTP (BF16 head, 2 drafts)**

| config | tok/s | peak wired |
|---|---|---|
| `--cpu-moe`, no MTP | 14.89 | 21.1 GiB |
| `--cpu-moe` + MTP | 21.48 | 21.8 GiB |
| `--n-cpu-moe 44` + MTP | **25.71** | 29.2 GiB |
| `--n-cpu-moe 40` + MTP | 22.21 | 39.2 GiB |
| `--n-cpu-moe 32` + MTP | 18.46 | 61.0 GiB |

**Window D, clean host, copy mode, Q8_0 MTP head unless noted**

| config | tok/s | peak wired |
|---|---|---|
| `--n-cpu-moe 44` + BF16 MTP 2 | 21.70 | 29.2 GiB |
| `--n-cpu-moe 44` + Q8_0 MTP 2 | 22.83 | 27.6 GiB |
| `--n-cpu-moe 45` + Q8_0 MTP 2 | 22.96 | 24.1 GiB |
| `--n-cpu-moe 46` / 43 / 42 + MTP 2 | 19.96 / 20.75 / 21.62 | 21.5-33.2 GiB |
| `--n-cpu-moe 44` + MTP 1 / 3 | 18.20 / 22.10 | ~27 GiB |

Same prompt repeated (all touched experts cached): copy `--cpu-moe` 22.4-23.8 tok/s, plus MTP
33.5-35.4 tok/s. That is the ceiling once experts never miss page cache.

## What it means

- MTP only helps once experts are in RAM. Streaming from SSD it cost 16-33%, because verifying
  drafts touches more experts. With experts cached it gave +44%.
- More GPU layers is slower past a point. Wired memory and page cache come from the same RAM,
  so pinning more expert layers starves the cache for the rest.
- The model needs ~130 GiB to keep every expert resident, slightly more than the machine has.
  The 22-26 tok/s band is the varied-prompt ceiling of this page-cache path on 128 GiB. It is not
  a ceiling for the machine: the 2026-10-10 TensorFold SSD expert pool measured 23.0-27.2 tok/s on
  new prompts from a cold cache with a 60 GiB pool (see the README).

## Prefill

10k-token prompt, copy mode + MTP, 256k context:

| setting | prefill |
|---|---|
| `-ub 512` (default) | ~5 tok/s |
| `-ub 2048` | 18-19 tok/s |
| `-ub 2048 --no-op-offload` | 31-32 tok/s |

`--no-op-offload` keeps CPU-resident experts on the CPU during batches instead of copying them
to the GPU for every batch.

## 256k context

Works. KV for 256k is ~9 GiB (12 of 48 layers hold KV). Server footprint ~32 GiB with
`-ub 2048`. Decode drops with depth: ~8 tok/s on short context and ~3.6 tok/s at 10k tokens
with other services running. Filling 256k takes ~2.2 h at 32 tok/s prefill.

## Did not help

- Async expert prefetch on CPU (`madvise` per routed expert): slower.
- 4 or 8 concurrent streams while streaming from SSD: no gain in total tokens.
- 18 threads (6 extra cores): 14 tok/s vs 22 at 12.
- `-ngl 26`/`28` contiguous: slower than `-ngl 24`.

## MTP head

The BF16 MTP head already in the HF repo loads on v0.6.0 (it has `nextn.enorm` and
`nextn.hnorm`). The Q8_0 head (`mtp-Qwen3.8-Flash-Next-Heretic-Q8_0.gguf`, 4.1 GB) is the
same tensors quantized with `llama-quantize`; it saves ~3.5 GiB for page cache and measured
equal or slightly faster.
