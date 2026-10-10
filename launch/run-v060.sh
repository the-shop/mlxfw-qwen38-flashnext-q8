#!/bin/bash
# MLXFW Q8_0 on llama.cpp v0.6.0 + copy-mode patch (the-shop/llama.cpp, branch mlxfw-copy-mode).
# Needs a warm page cache. This command measured 22.96 tok/s on varied prompts with 24 GiB wired on a 128 GiB M5 Max
# (2026-10-06, other RAM users stopped; best seen 25.7 at --n-cpu-moe 44). From a cold page cache: 2.3 tok/s.
set -eu
: "${LLAMA_DIR:?set LLAMA_DIR to your llama.cpp build (the-shop/llama.cpp @ mlxfw-copy-mode)}"
: "${MODEL_DIR:?set MODEL_DIR to the folder with q8-0000x-of-00005.gguf and the MTP head}"
CTX="${CTX:-65536}"          # 262144 works (+~9 GiB KV), see docs
NCPUMOE="${NCPUMOE:-45}"     # 44-45 measured best; lower = more GPU experts = less page cache
MTP="${MTP:-$MODEL_DIR/mtp-Qwen3.8-Flash-Next-Heretic-Q8_0.gguf}"
export LLAMA_GPU_NO_HOST_PTR=1
exec "$LLAMA_DIR/build/bin/llama-server" \
  -m "$MODEL_DIR/q8-00001-of-00005.gguf" \
  --no-repack -ngl 99 --n-cpu-moe "$NCPUMOE" -t 12 \
  -c "$CTX" -np 1 -b 2048 -ub 2048 --no-op-offload \
  --spec-type draft-mtp -md "$MTP" --spec-draft-n-max 2 \
  --jinja --host 127.0.0.1 --port "${PORT:-8130}"
