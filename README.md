# MLXFW — running a 188 GB Q8 MoE on one Apple Silicon Mac

Measurement harness and findings from fitting **Qwen3.8-Flash-Next Q8_0** (122B total /
10B active, 512 experts) onto a single M5 Max with 128 GiB of unified memory, while keeping
tens of gigabytes of RAM free for other work. MLXFW = "for work".

Weights: <https://huggingface.co/the-shop/MLXFW-Qwen3.8-Flash-Next-Heretic-Q8_0>

The original results below were measured on the **closed llama.cpp PR #27739** (`JJJYmmm`,
branch `add_qwen4exp` @ `dfa0c0f`). The 2026-10-06 update runs on upstream **llama.cpp v0.6.0**
plus a one-line copy-mode patch, and is the recommended way to run it now.

## Run it today (llama.cpp v0.6.0, 2026-10-06)

Best measured: **25.7 tok/s on varied prompts with 29 GiB wired** (typical 21-23 across
measurement windows), copy mode + MTP, M5 Max 128 GiB with other RAM users stopped.

```bash
git clone -b mlxfw-copy-mode https://github.com/the-shop/llama.cpp && cd llama.cpp
cmake -B build -DGGML_METAL=ON && cmake --build build -j --target llama-server
hf download the-shop/MLXFW-Qwen3.8-Flash-Next-Heretic-Q8_0 --local-dir ~/mlxfw \
  --include "q8-*.gguf" "mtp-Qwen3.8-Flash-Next-Heretic-Q8_0.gguf"
LLAMA_DIR=$PWD MODEL_DIR=~/mlxfw ../mlxfw-qwen38-flashnext-q8/launch/run-v060.sh
```

What the flags do:

- `LLAMA_GPU_NO_HOST_PTR=1` (the patch): the GPU copies its own tensors instead of wiring the
  whole mmap range of each file. Upstream PR: <https://github.com/ggml-org/llama.cpp/pull/30060>. Dense-on-GPU went from Metal OOM at 69-92 GiB wired to a
  4.85 GiB GPU buffer.
- `-ngl 99 --n-cpu-moe 45`: all dense weights plus 3 expert layers on GPU, the other experts
  stream from page cache. More GPU expert layers starves the cache and gets slower.
- `--spec-type draft-mtp` with the Q8_0 MTP head, 2 drafts: +44% once experts are cached.
  MTP hurts if experts stream from SSD.
- `-ub 2048 --no-op-offload`: prefill 5 -> 31 tok/s on a 10k prompt.
- `-t 12`: performance cores on M5 Max. 18 threads collapsed to 14 tok/s.

Free RAM matters more than any flag: ~7 tok/s with other services holding ~40 GiB, 22-26 with
them stopped. 256k context works (`CTX=262144`, +~9 GiB). Full tables and negative results:
`docs/2026-10-06-v060-copy-mode-mtp.md`.

## Original result (PR #27739 fork, 2026-10-04)

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

## Seven approaches that did not work

| approach | outcome | why |
|---|---|---|
| per-expert GPU/CPU placement | impossible as shipped | all 512 experts of a layer are one tensor; a tensor has one buffer; routing is decided in-graph so residency cannot follow the router |
| CPU/GPU compute overlap | patchable but unsafe | `ggml_gallocr` reuses scratch across splits assuming serial execution — concurrent splits alias buffers and give silently wrong output, no crash |
| MTLIO / sparse heaps | no gain | 12.0–13.5 GB/s matches the existing read path, pollutes page cache, placement-sparse is Private-only so no CPU fallback |
| `GGML_METAL_NO_RESIDENCY=1` | no-op | 45.1 vs 45.2 GiB wired; wiring comes from `newBufferWithBytesNoCopy`, not residency sets |
| speculative decoding | no gain | 20.73 / 20.84 / 20.68 at draft depth 4 / 8 / 16 against a 20.62 baseline |
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

Base model **Qwen Team, Alibaba** (`Qwen/Qwen3.8-Flash-Next`). Decensoring tool **p-e-w**
(`heretic`), fork **timrohrbaugh** v1.3.0+custom seed `2185752647`, weights **trohrbaugh**
(`trohrbaugh/Qwen3.8-Flash-Next-heretic`). Runtime **ggml-org** and llama.cpp contributors.
Architecture support merged as PR **#27742** (**unslothai**); every measurement here ran on
the closed PR **#27739** (**JJJYmmm**) — a closed PR is still someone's work, and it is the
branch that produced and served these files.
Stock upstream llama.cpp at the #27742 merge (`6c84c7d5`) loads the Q8_0 with `--no-repack`.
Its greedy output matches the fork byte for byte at `-ngl 6` but diverges early at `-ngl 12`/`16`
(both deterministic, both coherent), so the two are not numerically equivalent; the speed
figures are fork-only.

Sibling release and the full provenance chain:
<https://github.com/the-shop/qwen38-flashnext-hybrid-recipe>

Licence: Qwen Community License 1.0, inherited unchanged. Clause 2 — commercial
Model-as-a-Service serving needs a separate licence from Qwen, obtained beforehand.
