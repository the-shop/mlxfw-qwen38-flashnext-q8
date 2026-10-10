# Changelog

## v0.1.0 — 2026-10-10

First tagged release.

- TensorFold with an SSD expert pool is the recommended path: the MLX 8-bit checkpoint
  (`the-shop/MLXFW-Qwen3.8-Flash-Next-Heretic-MLX-8bit`) on the-shop/TensorFold v0.1.0.
  Measured 2026-10-10 on an M5 Max 128 GB with two harnesses and identical output tokens:
  60 GiB pool (`LIMIT=76`) 23.0-27.2 tok/s on new prompts with a cold prefill, about 59 tok/s on
  reruns; 40 GiB pool (`LIMIT=56`) 18.8-22.0 tok/s; 70 GiB pool (native) decode -2% (harness B) /
  +5% (harness A) vs 60 GiB, prefill 9-15% lower.
- `launch/run-tensorfold.sh` defaults to `POOL=60 LIMIT=76`.
- The llama.cpp v0.6.0 copy-mode path is labelled warm page cache only: 22.96 tok/s with the
  shipped command, 2.3 tok/s from a cold page cache.
- Corrections made before the tag:
  - `README.md`: the model has 177B parameters in total, including a ~51B n-gram
    table (previously stated as 122B); 16 reader threads raised prefill 4-9% at 40 and 60 GiB
    (previously 4-11%) with decode within ±3% there and ±4% at 70 GiB, except harness B serial decode
    at 40 GiB (-10%) (previously "within ±4%"); at 70 GiB the native pool server decodes 2-6% and
    prefills 8-11% faster than the Python loop (previously 2-6% for both); 70 GiB prefill is 9-15%
    below 60 GiB (previously 10-15%); 70 GiB decode is stated as -2% (harness B) / +5% (harness A)
    vs 60 GiB (previously "no gain").
  - Attribution: SSD expert streaming is ashhart's work in TensorFold 0.3.6; the-shop's changes are
    the 8-bit kernels and converter, the native pool server and the opt-in prefetch, seed and
    expert-pack flags.
  - `docs/MEASUREMENTS.md`: the QD1 storage ceiling uses the mincore-verified figures
    (5.16 GB/s external, 8.28 GB/s internal) instead of 6.70 GB/s; the n-gram table page-cache
    figure states 0.19 GiB in the measured runs and 0.03-0.47 GiB across all probes.
  - Licensing: scripts and code are under the MIT License (`LICENSE-CODE`); the model weights stay
    under the Qwen Community License 1.0 (`LICENSE`).
  - `docs/MEASUREMENTS.md`: the saturated storage ceilings use the mincore-verified figures
    (7.08 GB/s external, 13.52 GB/s internal); the speculative-decoding result is scoped to the
    PR #27739 branch; the M5 Max core layout (6 + 12 cores) is stated correctly.
  - Attribution links upstream projects and PR authors.
- Repository rules for releases, authorship and published numbers.
