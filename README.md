# MLXFW — running a 188 GB Q8 MoE on one Apple Silicon Mac

Measurement harness and findings from fitting **Qwen3.8-Flash-Next** at 8 bits (122B total /
10B active, 512 experts) onto a single M5 Max with 128 GiB of unified memory, while keeping
tens of gigabytes of RAM free for other work. MLXFW = "for work".

Weights:

- MLX 8-bit, for TensorFold (recommended): <https://huggingface.co/the-shop/MLXFW-Qwen3.8-Flash-Next-Heretic-MLX-8bit>
- GGUF Q8_0, for llama.cpp: <https://huggingface.co/the-shop/MLXFW-Qwen3.8-Flash-Next-Heretic-Q8_0>

Three generations of results are below, newest first: TensorFold with an SSD expert pool
(2026-10-10, recommended), llama.cpp v0.6.0 with a copy-mode patch (2026-10-06, needs a warm
page cache), and the original GPU/CPU tiering study on the closed llama.cpp PR #27739 branch
(2026-10-04).

## Recommended: TensorFold 8-bit with an SSD expert pool (2026-10-10)

The same model at 8-bit, run on [TensorFold](https://github.com/the-shop/TensorFold/tree/v0.1.0)
(the-shop's fork of [ashhart/TensorFold](https://github.com/ashhart/TensorFold) v0.3.6.3, release v0.1.0)
instead of llama.cpp. The experts stay on the internal SSD and a fixed-size pool in RAM holds the
ones in use, so this path doesn't depend on the page cache. It needs an M5 Mac with 128 GB.

| `POOL` / `LIMIT` (GiB) | Pool reader | New prompt, drafts on | Same prompt rerun | Prefill, cold | Peak (MLX) | Pool hits |
|---|---|---|---|---|---|---|
| 40 / 56 (light) | native (default) | 18.8-22.0 tok/s | 25.0 tok/s | 198-263 tok/s | 52 GiB | 90% |
| 40 / 56 | Python loop | 16.9-20.0 tok/s | 23.4 tok/s | 183-244 tok/s | 52 GiB | 90% |
| **60 / 76 (recommended)** | **native (default)** | **23.0-27.2 tok/s** | **58.8 tok/s** | **190-255 tok/s** | **72 GiB** | **98%** |
| 60 / 76 | Python loop | 20.7-24.9 tok/s | 59.5 tok/s | 176-240 tok/s | 72 GiB | 98% |
| 70 / 86 (no gain) | native (default) | 22.5-28.6 tok/s | 60.0 tok/s | 172-217 tok/s | 82 GiB | 98% |
| 70 / 86 | Python loop | 22.0-27.1 tok/s | 58.1 tok/s | 160-195 tok/s | 82 GiB | 98% |

- **Which pool:** 60 GiB with `LIMIT=76` if you can give the server about 75 GiB. 40 GiB with
  `LIMIT=56` leaves about 75 GiB free for other work. On new prompts 60 GiB is 22-24% faster than
  40 GiB. 70 GiB gave no gain over 60, with 10-15% lower prefill, and needs 10 GiB more.
- **Reruns:** at 60 and 70 GiB a whole reply's experts stay resident, so a repeat of the same prompt
  runs at about 59 tok/s. At 40 GiB a repeat reaches 25 tok/s.
- **Pool hits:** the share of expert reads served from the pool: 90% at 40 GiB, 98% at 60 and 70 GiB.
  Expert reads that miss the pool, and n-gram reads, go to the SSD with `F_NOCACHE`, bypassing the
  page cache.
- **Native pool server (default):** a C++ thread in TensorFold's host-sync extension serves each
  layer's expert reads. Against the Python loop (`TENSORFOLD_POOL_NATIVE=0`) it decodes 9-11% faster
  with drafts and prefills 6-8% faster at 40 and 60 GiB, 2-6% at 70 GiB. Output tokens are identical.
- **Did not help:** 16 reader threads (`TENSORFOLD_POOL_THREADS=16`) raised prefill 4-11% at 40 and
  60 GiB but not at 70, with decode within ±4%, so the default stays 8. The contiguous expert pack
  (`TENSORFOLD_EXPERT_PACK=1`) decoded 16-23% slower at 40 and 60 GiB and is opt-in and experimental.
- **Identical tokens:** output tokens were identical across all 12 measured configurations (three pool
  sizes x native, Python loop, 16 threads and pack), with drafts on and off.
- **How it was measured:** M5 Max, 128 GB, internal SSD, `--context 16384`, `--ple-on-ssd`,
  `--no-thinking`, temperature 0, replies of up to 128 tokens, prompt snapshots off so every prefill
  is cold, page cache displaced before each window. Other model servers and background jobs were
  stopped, and swap stayed flat during the 60 and 70 GiB runs. Two independent harnesses; each value
  is a median of server-side tok/s. The low end of each range is harness B (7 short prompts, first
  visit, drafts on), the high end harness A (5 prompts, first run of each, drafts on). Reruns are
  harness A runs 2-5. Prefill ranges are harness B (2,475-token prompt) to harness A (3,381-token
  prompt).

Run it from a fresh clone. You need Xcode Command Line Tools, Python 3.11 or later as `python3`,
and about 200 GB of free SSD:

```bash
git clone https://github.com/the-shop/mlxfw-qwen38-flashnext-q8.git
git clone -b v0.1.0 https://github.com/the-shop/TensorFold.git && cd TensorFold
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[ssd]" "huggingface_hub>=0.34"
hf download the-shop/MLXFW-Qwen3.8-Flash-Next-Heretic-MLX-8bit --local-dir ~/mlxfw-q8   # 194 GB
POOL=60 LIMIT=76 MODEL_DIR=~/mlxfw-q8 ../mlxfw-qwen38-flashnext-q8/launch/run-tensorfold.sh
```

Use `POOL=40 LIMIT=56` to leave more RAM free. The first start builds the host-sync extension with
cmake and nanobind. The server listens on `127.0.0.1:8131`; `curl -s 127.0.0.1:8131/health` shows
`"native": 1` under `experts` when the native pool server is in use. A larger `LIMIT` buys context:
with a 40 GiB pool, `LIMIT=62` starts and serves `CTX=65536`.

To build the 8-bit checkpoint yourself, convert the 360 GB BF16 source (about 560 GB of free SSD)
from the TensorFold checkout with
`python tools/convert_flash_next_q8.py <BF16 dir> flashnext-q8 --limit-gb 14`. To start from the
original, non-decensored model, download `Qwen/Qwen3.8-Flash-Next` as the BF16 source. The full
recipe covers context and memory trade-offs, the token-identity check for drafting, and the opt-in
prefetch and seed flags:
[`docs/recipes/qwen3.8-flash-next.md`](https://github.com/the-shop/TensorFold/blob/v0.1.0/docs/recipes/qwen3.8-flash-next.md#running-it-on-a-128-gb-m5-mac).

## llama.cpp v0.6.0 + copy mode, warm page cache only (2026-10-06)

The shipped command (`--n-cpu-moe 45`, Q8_0 MTP head, 2 drafts) measured **22.96 tok/s with
24 GiB wired** on varied prompts; the best seen was 25.7 tok/s at `--n-cpu-moe 44`. These figures
need the page cache to hold most experts. From a cold page cache the same setup measured
**2.3 tok/s** (2026-10-07, two independent harnesses), which is why the TensorFold path above is
recommended for new prompts.

```bash
git clone -b mlxfw-copy-mode https://github.com/the-shop/llama.cpp && cd llama.cpp
cmake -B build -DGGML_METAL=ON && cmake --build build -j --target llama-server
hf download the-shop/MLXFW-Qwen3.8-Flash-Next-Heretic-Q8_0 --local-dir ~/mlxfw \
  --include "q8-*.gguf" "mtp-Qwen3.8-Flash-Next-Heretic-Q8_0.gguf"
LLAMA_DIR=$PWD MODEL_DIR=~/mlxfw ../mlxfw-qwen38-flashnext-q8/launch/run-v060.sh
```

What the flags do:

- `LLAMA_GPU_NO_HOST_PTR=1` (the patch): the GPU copies its own tensors instead of wiring the
  whole mmap range of each file. It is proposed upstream as a draft PR,
  <https://github.com/ggml-org/llama.cpp/pull/30060>. Dense-on-GPU went from Metal OOM at
  69-92 GiB wired to a 4.85 GiB GPU buffer.
- `-ngl 99 --n-cpu-moe 45`: all dense weights plus 3 expert layers on GPU, the other experts
  stream from page cache. More GPU expert layers starves the cache and gets slower.
- `--spec-type draft-mtp` with the Q8_0 MTP head, 2 drafts: +44% once experts are cached.
  MTP hurts if experts stream from SSD.
- `-ub 2048 --no-op-offload`: prefill 5 -> 31 tok/s on a 10k prompt.
- `-t 12`: 12 threads. The M5 Max has 18 CPU cores, 6 at its top performance level and 12 at the
  next (`sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu`). 18 threads collapsed to
  14 tok/s.

Free RAM matters more than any flag: ~7 tok/s with other services holding ~40 GiB, 22-26 with
them stopped. 256k context works (`CTX=262144`, +~9 GiB). Full tables and negative results:
`docs/2026-10-06-v060-copy-mode-mtp.md`.

## Original result (PR #27739 branch, 2026-10-04)

| `-ngl` | tok/s | wired | RAM left free |
|---|---|---|---|
| 6 | 9.7 | 19.4 GiB | ~60 GiB |
| **16** | **20.45** | 45.1 GiB | **39 GiB** |
| 24 | 23.35 | 65.6 GiB | 28 GiB |
| 28 | 22.36 | 76.0 GiB | 22 GiB |

12 varied prompts per config, 24 warm-up queries discarded, token-weighted, outputs checked
non-empty.

## Two rules that transfer to any Mac

**1. Put a contiguous block of layers on the GPU. Never cherry-pick expert tensors.**
At equal GPU memory (45.0 vs 46.8 GiB) contiguous `-ngl 16` beats scattered per-layer expert
placement through `-ot` by **+28.6%** — 20.80 vs 16.18 tok/s. Each CPU↔GPU transition costs a
full GPU drain plus host-side tensor copies, and scattered placement pays two per *layer*
rather than two per *token*.

**2. Metal charges file distance, not bytes.** A file's GPU buffer spans from the first to the
last byte it needs, so one GPU-assigned tensor near the start of a file stretches the span
across the whole file. Dense weights for layers 0–31 are 2.8 GiB of data but **79 GiB of span**.

`tier/span.py` predicts this from GGUF headers and refuses unsafe placements:

```
python3 tier/span.py "ple_ngram_embd=CPU,per_layer_token_embd=CPU,token_embd=CPU" 16 <model.gguf>
```

The bound is total span against `recommendedMaxWorkingSetSize` (120 GiB on this machine).
`maxBufferLength` is **not** a limit — ggml splits oversized spans into overlapping views
(`ggml-metal-device.m:1753-1780`), so 80.64 GiB is a view stride. An earlier version of this
tool enforced it as a cliff and rejected valid configurations.

## Under memory pressure, a bigger GPU tier wins

Counterintuitive and measured. GPU-resident weights are wired and cannot be evicted, so more
weights on the GPU means less dependence on page cache that other work keeps stealing.

| tier | held by other work | tok/s |
|---|---|---|
| `-ngl 24` | 15 GiB | 10.29 |
| `-ngl 16` | 30 GiB | 6.61 |
| `-ngl 6` | 45 GiB | 5.75 |

Budget: `wired + held + ~36 GiB working cache ≤ 122 GiB`. So "60 GiB free" and a large hot
tier are mutually exclusive on 128 GiB.

## The harness

`tier/gate.sh` runs one configuration with a fail-closed memory guard; `tier/verdict.sh`
decides PASS/INVALID from the artifacts, including that the memory hold **spanned the whole
decode window** rather than being established and released around it. `tier/hog2.py` holds a
given number of GiB of *incompressible* memory and aborts on the first swapout.

```
bash tier/tests/run_tests.sh      # 11 tests, no model needed, seconds
```

The tests exist because two earlier harness versions produced results that looked valid and
were not: one continued measuring after the memory hold had died, and one defaulted its
swapout reader to 0 when the field was missing. Both would have reported clean runs.

## Seven approaches that did not work (2026-10-04, PR #27739 branch)

| approach | outcome | why |
|---|---|---|
| per-expert GPU/CPU placement | impossible as shipped | all 512 experts of a layer are one tensor; a tensor has one buffer; routing is decided in-graph so residency cannot follow the router |
| CPU/GPU compute overlap | patchable but unsafe | `ggml_gallocr` reuses scratch across splits assuming serial execution — concurrent splits alias buffers and give silently wrong output, no crash |
| MTLIO / sparse heaps | no gain | 12.0–13.5 GB/s matches the existing read path, pollutes page cache, placement-sparse is Private-only so no CPU fallback |
| `GGML_METAL_NO_RESIDENCY=1` | no-op | 45.1 vs 45.2 GiB wired; wiring comes from `newBufferWithBytesNoCopy`, not residency sets |
| speculative decoding on the #27739 branch | no gain there | 20.73 / 20.84 / 20.68 at draft depth 4 / 8 / 16 against a 20.62 baseline. Later, MTP drafting on llama.cpp v0.6.0 gave +44% with experts cached (see above) |
| suppressing n-gram readahead | premise false | the 50 GB table holds 0.19 GiB of page cache, not the 3–7 GiB predicted. Patch written, built, measured, no change |
| weights on internal NVMe | ~9%, unreplicated | internal is 1.91× faster (13.52 vs 7.08 GB/s) yet gave only +9% at a 30 GiB hold, n=1. mmap faulting caps at ~1.5 GB/s per thread, so the fault path is the ceiling, not the device |

## Measuring storage correctly

`F_NOCACHE` stops *new* cache insertion but does nothing about pages already resident. Two
benchmarks here reported 22 and then 94 GB/s before that was noticed — the second above
Thunderbolt line rate, which should have been the immediate tell. Valid figures require
checking each offset with `mincore` and timing only non-resident ones:

```
external TBT5 SSD   5.16 GB/s @ QD1,  7.08 saturated
internal NVMe       8.28 GB/s @ QD1, 13.52 saturated
```

## Full detail

`docs/MEASUREMENTS.md` — every configuration, the closed paths with source citations, and the
corrections made along the way.

## Attribution

- Base model: **Qwen Team, Alibaba**, [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
- Decensoring tool: [heretic](https://github.com/p-e-w/heretic) by **p-e-w**; run from the
  timrohrbaugh/heretic fork, v1.3.0+custom, seed `2185752647` (the fork's GitHub page returned 404 on
  2026-10-10, so it is not linked). Weights: **trohrbaugh**,
  [`trohrbaugh/Qwen3.8-Flash-Next-heretic`](https://huggingface.co/trohrbaugh/Qwen3.8-Flash-Next-heretic).
- llama.cpp runtime: [**ggml-org**](https://github.com/ggml-org/llama.cpp) and the llama.cpp contributors.
  Architecture support: [PR #27739](https://github.com/ggml-org/llama.cpp/pull/27739) by **JJJYmmm**
  (closed) and [PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742) by Daniel Han of
  **unslothai** (merged).
- TensorFold runtime: [**ashhart/TensorFold**](https://github.com/ashhart/TensorFold), built on
  [**MLX**](https://github.com/ml-explore/mlx) by ml-explore. The SSD expert pool, the native pool
  server and the 8-bit converter are the-shop's changes on top of ashhart/TensorFold
  [v0.3.6.3](https://github.com/ashhart/TensorFold/tree/v0.3.6.3) (MIT), released as
  [the-shop/TensorFold v0.1.0](https://github.com/the-shop/TensorFold/tree/v0.1.0).

Which code produced which numbers: the 2026-10-04 tiering study ran on the closed PR #27739 branch
(`add_qwen4exp` @ `dfa0c0f`), which produced and served the GGUF files; a closed PR is still someone's
work. The 2026-10-06 llama.cpp figures ran on llama.cpp v0.6.0 plus the copy-mode patch, and the
2026-10-10 figures on the-shop/TensorFold. Stock upstream llama.cpp at the #27742 merge (`6c84c7d5`)
loads the Q8_0 with `--no-repack`. Its greedy output matches the #27739 branch byte for byte at
`-ngl 6` but diverges early at `-ngl 12`/`16` (both deterministic, both coherent), so the two are not
numerically equivalent, and the 2026-10-04 speed figures apply to the #27739 branch only.

Sibling release and the full provenance chain:
<https://github.com/the-shop/qwen38-flashnext-hybrid-recipe>

Licence: Qwen Community License 1.0, inherited unchanged. Clause 2 — commercial
Model-as-a-Service serving needs a separate licence from Qwen, obtained beforehand.
