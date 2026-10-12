#!/bin/bash
# MLXFW Qwen3.8-Flash-Next-Heretic, MLX 8-bit, on TensorFold with an SSD expert pool
# (the-shop/TensorFold v0.3.0). Needs an M5 Mac with 128 GB of RAM.
#
# v0.3.0, measured 2026-10-12 on an M5 Max against TensorFold v0.2.0, same flags, two runs per build,
# output tokens identical on every request. POOL=60 LIMIT=84 CTX=77824 plus
# --parallel 1 --persist-turns --checkpoint-last-user, TF_PREFILL_LAYER_MAJOR=1, page cache evicted
# (server started without --no-thinking; agent-style requests with thinking on, greedy):
#
#   cold 24,015-token prompt, time to first token     33-39 s     (v0.2.0: 292-323 s)
#   next turn on the same warm server (medians)       2.9-3.4 s   (v0.2.0: 5.3-6.5 s)
#   peak process footprint                            87.9 GB     (v0.2.0: 87.1 GB; 2.3 GB under the
#                                                                  84 GiB = 90.19 GB budget)
#
# The 33-39 s cold figure needs TF_PREFILL_LAYER_MAJOR=1 (off by default in TensorFold v0.3.0, set below
# unless you set it); cold prefill with TensorFold's defaults was not measured. Output tokens are identical
# with it on or off. Extra arguments are passed to `tensorfold serve`.
#
# v0.1.0, measured 2026-10-10 on an M5 Max, native pool server, cold prefill, two harnesses
# (new prompts with drafts; prefill on 2,475- and 3,381-token prompts):
#
#   POOL  LIMIT  CTX     new prompts     prefill        peak (MLX)  pool hits
#   40    56     16384   18.8-22.0 tok/s 198-263 tok/s  52 GiB      90%   light; leaves ~75 GiB free
#   40    62     65536   n/a             n/a            n/a         n/a   longer chats; prompts over ~55K tokens re-prefill each turn
#   60    76     16384   23.0-27.2 tok/s 190-255 tok/s  72 GiB      98%   recommended (default); reruns ~59 tok/s
#   70    86     16384   22.5-28.6 tok/s 172-217 tok/s  82 GiB      98%   vs 60: decode -2% (B) / +5% (A), prefill 9-15% lower; needs 10 GiB more
#
# TENSORFOLD_POOL_NATIVE=0 selects the slower Python pool loop; output tokens are identical either way.
set -eu
: "${MODEL_DIR:?set MODEL_DIR to the folder from: hf download the-shop/MLXFW-Qwen3.8-Flash-Next-Heretic-MLX-8bit}"
POOL="${POOL:-60}"
LIMIT="${LIMIT:-76}"
CTX="${CTX:-16384}"
TENSORFOLD="${TENSORFOLD:-tensorfold}"   # or: "python -m tensorfold.cli" from the TensorFold checkout
export TENSORFOLD_MEMORY_LIMIT_GB="$LIMIT"
export TF_PREFILL_LAYER_MAJOR="${TF_PREFILL_LAYER_MAJOR:-1}"
exec $TENSORFOLD serve "$MODEL_DIR" --name mlxfw-q8 \
  --ssd-experts "$POOL" --ple-on-ssd --context "$CTX" --no-thinking \
  --host 127.0.0.1 --port "${PORT:-8131}" --no-update-check "$@"
