#!/bin/bash
# MLXFW Qwen3.8-Flash-Next-Heretic, MLX 8-bit, on TensorFold with an SSD expert pool
# (the-shop/TensorFold, branch q8-flash-next). Needs an M5 Mac with 128 GB of RAM.
# Measured 2026-10-09 on an M5 Max: 40 GiB pool -> 15.6-18.8 tok/s on new prompts, 52 GiB peak.
#
#   POOL  LIMIT  CTX     peak (MLX)
#   40    56     16384   52 GiB    default; leaves the most RAM for other work
#   40    62     65536   n/a       longer chats; prompts over ~55K tokens re-prefill each turn
#   60    76     16384   72 GiB    ~20% faster on new prompts, far faster on repeated ones
set -eu
: "${MODEL_DIR:?set MODEL_DIR to the folder from: hf download the-shop/MLXFW-Qwen3.8-Flash-Next-Heretic-MLX-8bit}"
POOL="${POOL:-40}"
LIMIT="${LIMIT:-56}"
CTX="${CTX:-16384}"
TENSORFOLD="${TENSORFOLD:-tensorfold}"   # or: "python -m tensorfold.cli" from the TensorFold checkout
export TENSORFOLD_MEMORY_LIMIT_GB="$LIMIT"
exec $TENSORFOLD serve "$MODEL_DIR" --name mlxfw-q8 \
  --ssd-experts "$POOL" --ple-on-ssd --context "$CTX" --no-thinking \
  --host 127.0.0.1 --port "${PORT:-8131}" --no-update-check
