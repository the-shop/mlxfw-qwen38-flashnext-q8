#!/bin/bash
# Pure verdict over run artifacts. Exit 0 ONLY for PASS. No model, no side effects,
# so it is directly unit-testable against crafted fixtures.
# usage: verdict.sh <hoglog> <events.tsv> <srvlog> <resp_prefix> <expected_hold_gib> <n_gens>
HOGLOG=$1; EV=$2; SRVLOG=$3; RP=$4; WANT=$5; NGENS=${6:-5}
fail(){ echo "INVALID: $1"; exit 1; }
for f in "$HOGLOG" "$EV"; do [ -s "$f" ] || fail "missing/empty artifact: $f"; done
[ -f "$SRVLOG" ] || fail "missing server log: $SRVLOG"

grep -q "ABORT_GUARD" "$HOGLOG" && fail "hog tripped its guard: $(grep -m1 'RELEASED (' "$HOGLOG")"
grep -q "MemoryError" "$HOGLOG" && fail "hog hit MemoryError"
HELD=$(sed -nE 's/.*HOLDING=([0-9]+) GiB.*/\1/p' "$HOGLOG" | tail -1)
[ -n "$HELD" ] || fail "hog never reached HOLDING"
[ "$HELD" -ge "$WANT" ] || fail "held ${HELD} GiB < required ${WANT} GiB"

# authoritative no-new-swap signal: raw vm_stat Swapouts delta must be exactly 0
SO=$(sed -nE 's/.*swapouts=\+([0-9]+).*/\1/p' "$HOGLOG" | sort -rn | head -1)
[ -n "$SO" ] || fail "no swapouts reading in hog log (reader must never default)"
[ "$SO" -eq 0 ] || fail "swapouts grew by $SO during the run"

# the hold must SPAN the whole decode window, else it proves nothing about concurrency
T_HOLD=$(awk -F'\t' '$2=="hog_holding"{print $1}' "$EV")
T_FIRST=$(awk -F'\t' '/decode_start_hold/{print $1; exit}' "$EV")
T_DONE=$(awk -F'\t' '$2=="gen_done"{print $1}' "$EV")
T_REL=$(sed -nE 's/^\[([0-9.]+)\] RELEASED.*/\1/p' "$HOGLOG" | tail -1)
for v in T_HOLD T_FIRST T_DONE T_REL; do [ -n "${!v}" ] || fail "missing timestamp $v"; done
awk -v a="$T_HOLD" -v b="$T_FIRST" 'BEGIN{exit !(a<=b)}' || fail "hold started after decode began"
awk -v a="$T_DONE" -v b="$T_REL"   'BEGIN{exit !(a<=b)}' || fail "hog released before generation finished"

# every generation must have completed and returned usable content
for i in $(seq 1 "$NGENS"); do
  awk -F'\t' -v t="decode_end_hold$i" '$2==t{f=1} END{exit !f}' "$EV" || fail "gen $i never completed"
  J="${RP}hold$i"*.json; J=$(ls $J 2>/dev/null | head -1)
  [ -s "$J" ] || fail "gen $i produced no response file"
  python3 - "$J" 2>/dev/null <<'PY' || fail "gen $i response unusable"
import json,sys
d=json.load(open(sys.argv[1])); c=d['choices'][0]
assert (c['message'].get('content') or '').strip(), 'empty content'
assert c['finish_reason']=='stop', f"finish_reason={c['finish_reason']}"
PY
done

SPAN=$(grep -c 'recommended max working set' "$SRVLOG"); OOM=$(grep -c 'OutOfMemory' "$SRVLOG")
[ "$SPAN" -eq 0 ] || fail "$SPAN Metal span warnings"
[ "$OOM" -eq 0 ] || fail "$OOM Metal OOM"
echo "PASS: held=${HELD}GiB (>=${WANT}) swapouts=+0 gens=$NGENS span=0 oom=0 hold spanned decode window"
