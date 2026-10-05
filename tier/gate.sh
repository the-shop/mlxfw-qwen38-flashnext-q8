#!/bin/bash
# Concurrency gate: model serving on GPU at ngl=N while an incompressible HOLD of
# HOG GiB stays resident for the WHOLE decode window. Fail-closed: if the hog dies,
# trips its guard, or the server dies, the run is abandoned and reported INVALID.
# Stub-testable: BIN, HOG_PY, PORT, NGENS may be overridden from the environment.
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NGL=${1:-22}; HOG=${2:-60}
BIN=${BIN:?set BIN to your llama-server path}
M=${M:?set M to shard q8-00001-of-00005.gguf}
D=${OUTDIR:-./out}
HOG_PY=${HOG_PY:-$SD/hog2.py}; PORT=${PORT:-8097}; TAG=${TAG:-$NGL-$HOG}
PREFETCH=${PREFETCH:-0}; THREADS=${THREADS:-}; EXTRA=${EXTRA:-}
CTX=${CTX:-4096}; CRAM=${CRAM:-512}
THREAD_ARG=""; [ -n "$THREADS" ] && THREAD_ARG="--threads $THREADS"
SLOG=$D/gate-srv-$TAG.log; HLOG=$D/gate-hog-$TAG.log
SMP=$D/gate-samples-$TAG.tsv; EV=$D/gate-events-$TAG.tsv
rm -f /tmp/hogstop.$TAG /tmp/smpstop.$TAG "$SMP" "$EV" "$HLOG"
pgrep -f "$BIN" >/dev/null && { echo "INVALID: a server is already running"; exit 1; }
ev(){ echo -e "$(date +%s.%N)\t$1" >> "$EV"; }
SRV=""; HOGPID=""; SMPID=""
cleanup(){ touch /tmp/hogstop.$TAG /tmp/smpstop.$TAG 2>/dev/null; /bin/sleep 2
           [ -n "$SRV" ] && kill $SRV 2>/dev/null; [ -n "$SMPID" ] && kill $SMPID 2>/dev/null
           [ -n "$HOGPID" ] && kill $HOGPID 2>/dev/null; /bin/sleep 3; }
abort(){ echo "INVALID: $1"; cleanup; exit 2; }
trap 'abort "interrupted"' INT TERM

LLAMA_MMAP_PREFETCH=$PREFETCH "$BIN" --model "$M" --host 127.0.0.1 --port $PORT $THREAD_ARG $EXTRA \
  --ctx-size $CTX --parallel 1 --n-gpu-layers "$NGL" --fit off --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' --reasoning-format auto \
  --temp 0 --top-k 1 --cache-ram $CRAM --predict 256 --metrics --no-warmup --no-repack \
  -ot "ple_ngram_embd=CPU,per_layer_token_embd=CPU,token_embd=CPU" > "$SLOG" 2>&1 &
SRV=$!; ev "server_spawn ngl=$NGL"
OK=0; for i in $(seq 1 90); do curl -fsS http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { OK=1; break; }; kill -0 $SRV 2>/dev/null || break; /bin/sleep 4; done
[ $OK -ne 1 ] && abort "server never became healthy"
ev server_ready

( echo -e "epoch\tavail_gib\twired_gib\tswap_mb\tcompressor_gib" > "$SMP"
  while [ ! -f /tmp/smpstop.$TAG ]; do
    vm_stat | awk -v t="$(date +%s.%N)" -v sw="$(sysctl -n vm.swapusage | sed -nE 's/.*used = ([0-9.]+)M.*/\1/p')" \
      '/page size/{p=$8}/Pages free/{gsub(/\./,"",$3);f=$3}/Pages inactive/{gsub(/\./,"",$3);i=$3}/wired down/{gsub(/\./,"",$4);w=$4}/occupied by compressor/{gsub(/\./,"",$5);c=$5}
       END{printf "%s\t%.2f\t%.2f\t%s\t%.2f\n",t,(f+i)*p/1073741824,w*p/1073741824,sw,c*p/1073741824}' >> "$SMP"
    /bin/sleep 1
  done ) & SMPID=$!

# fail-closed supervision: the hold and the server must BOTH still be live
hog_ok(){ [ -n "$HOGPID" ] && kill -0 "$HOGPID" 2>/dev/null \
          && ! grep -q 'ABORT_GUARD\|MemoryError\|RELEASED' "$HLOG"; }
srv_ok(){ kill -0 $SRV 2>/dev/null; }

gen(){ # $1=tag $2=json-quoted prompt
  srv_ok || abort "server died before gen $1"
  [ "$1" = warmup ] || hog_ok || abort "hold not live before gen $1: $(tail -1 "$HLOG")"
  ev "decode_start_$1"
  curl -fsS --max-time 900 http://127.0.0.1:$PORT/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":$2}],\"n_predict\":128,\"temperature\":0}" \
    > "$D/gate-resp-$1-$TAG.json" 2>&1 || abort "request failed for gen $1"
  ev "decode_end_$1"
  srv_ok || abort "server died during gen $1"
  [ "$1" = warmup ] || hog_ok || abort "hold DIED during gen $1: $(tail -1 "$HLOG")"
}

gen warmup '"Say the word ready."'   # run 1 is always a cold-cache artifact, discarded
ev warm_done

( python3 "$HOG_PY" "$HOG" /tmp/hogstop.$TAG 256 > "$HLOG" 2>&1 ) & HOGPID=$!
for i in $(seq 1 180); do
  grep -q 'HOLDING=' "$HLOG" && break
  grep -q 'ABORT_GUARD\|MemoryError\|Traceback' "$HLOG" && break
  kill -0 $HOGPID 2>/dev/null || break
  /bin/sleep 2
done
grep -q 'HOLDING=' "$HLOG" || abort "hold never reached ${HOG} GiB: $(tail -2 "$HLOG" | tr '\n' ' ')"
ev hog_holding
hog_ok || abort "hold died immediately after reporting HOLDING"

gen hold1 '"What is the capital city of Australia? Reply with the city name only."'
gen hold2 '"List the first five prime numbers, separated by spaces, and nothing else."'
gen hold3 '"Explain in exactly three sentences why the sky appears blue."'
gen hold4 '"Name the four largest planets in our solar system, largest first, one per line."'
gen hold5 '"Write a haiku about external SSDs."'
NGENS=5
ev gen_done
hog_ok || abort "hold died before generation finished"

touch /tmp/hogstop.$TAG; wait $HOGPID 2>/dev/null; HRC=$?
/bin/sleep 2; touch /tmp/smpstop.$TAG; /bin/sleep 2
kill $SRV $SMPID 2>/dev/null; /bin/sleep 4
[ $HRC -ne 0 ] && { echo "INVALID: hog exited rc=$HRC: $(tail -2 "$HLOG" | tr '\n' ' ')"; exit 2; }
# verdict is computed from artifacts by a separate, unit-tested script
"$SD/verdict.sh" "$HLOG" "$EV" "$SLOG" "$D/gate-resp-" "$HOG" "$NGENS"
RC=$?; echo "ngl=$NGL hog=${HOG}GiB verdict_rc=$RC"; exit $RC
