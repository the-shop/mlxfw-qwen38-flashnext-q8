#!/bin/bash
# Lightweight fail-closed supervision tests. No model, no GPU, seconds to run.
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); D=$(dirname "$T")
OUT=$T/out; rm -rf $OUT; mkdir -p $OUT
PASSN=0; FAILN=0
chk(){ # name expect_substr actual_output
  if echo "$3" | grep -q "$2"; then echo "  ok   $1"; PASSN=$((PASSN+1));
  else echo "  FAIL $1 :: wanted '$2' got: $(echo "$3" | tail -2 | tr '\n' ' ')"; FAILN=$((FAILN+1)); fi; }

run_gate(){ # $1=hogfile $2=tag  -> prints gate output; gate spawns the stub itself
  BIN=$T/stub_llama.py M=/dev/null OUTDIR=$OUT HOG_PY=$T/$1 PORT=8399 TAG=$2 \
    bash $D/gate.sh 6 60 2>&1
  pkill -f "$T/stub_llama.py" 2>/dev/null; sleep 0.5
}

echo "== gate.sh supervision =="
chk "healthy hold -> PASS"            "PASS: held=60GiB"  "$(run_gate fake_hog_ok.py t_ok)"
chk "hog dies mid-decode -> INVALID"  "INVALID"           "$(run_gate fake_hog_dies.py t_dies)"
chk "hog trips guard -> INVALID"      "INVALID"           "$(run_gate fake_hog_guard.py t_guard)"
chk "hold short of target -> INVALID" "INVALID"           "$(run_gate fake_hog_short.py t_short)"
chk "swapouts>0 -> INVALID"           "INVALID"           "$(run_gate fake_hog_swapouts.py t_swapouts)"

echo "== verdict.sh unit cases =="
mk(){ # build a minimal artifact set: $1=dir
  mkdir -p $1; H=$1/h.log; E=$1/e.tsv; S=$1/s.log; : > $S
  printf '[100.0] HOLDING=60 GiB nonzero_probe=60/60 swap=100 MiB\n[101.0] hold swap=100 (+0) swapouts=+0\n[120.0] RELEASED (stopfile) swap=100 MiB\n[120.1] DONE held=60 GiB swapouts=+0 swap=+0 MiB\n' > $H
  { echo -e "100.5\thog_holding"; for i in 1 2 3 4 5; do echo -e "10$i.0\tdecode_start_hold$i"; echo -e "10$i.5\tdecode_end_hold$i"; done; echo -e "119.0\tgen_done"; } > $E
  for i in 1 2 3 4 5; do echo '{"choices":[{"finish_reason":"stop","message":{"content":"x"}}]}' > $1/gate-resp-hold$i-t.json; done
}
mk $OUT/v_good
chk "clean artifacts -> PASS" "PASS:" "$(bash $D/verdict.sh $OUT/v_good/h.log $OUT/v_good/e.tsv $OUT/v_good/s.log $OUT/v_good/gate-resp- 60 5)"
mk $OUT/v_empty; echo '{"choices":[{"finish_reason":"stop","message":{"content":"   "}}]}' > $OUT/v_empty/gate-resp-hold3-t.json
chk "empty content -> INVALID" "INVALID" "$(bash $D/verdict.sh $OUT/v_empty/h.log $OUT/v_empty/e.tsv $OUT/v_empty/s.log $OUT/v_empty/gate-resp- 60 5)"
mk $OUT/v_early; sed -i '' 's/^\[120.0\] RELEASED/[105.0] RELEASED/' $OUT/v_early/h.log
chk "released before gen_done -> INVALID" "INVALID" "$(bash $D/verdict.sh $OUT/v_early/h.log $OUT/v_early/e.tsv $OUT/v_early/s.log $OUT/v_early/gate-resp- 60 5)"
mk $OUT/v_nosw; sed -i '' 's/swapouts=+0/swapouts=unknown/g' $OUT/v_nosw/h.log
chk "no swapouts reading -> INVALID" "INVALID" "$(bash $D/verdict.sh $OUT/v_nosw/h.log $OUT/v_nosw/e.tsv $OUT/v_nosw/s.log $OUT/v_nosw/gate-resp- 60 5)"
mk $OUT/v_span; echo 'ggml_metal: recommended max working set size exceeded' > $OUT/v_span/s.log
chk "Metal span warning -> INVALID" "INVALID" "$(bash $D/verdict.sh $OUT/v_span/h.log $OUT/v_span/e.tsv $OUT/v_span/s.log $OUT/v_span/gate-resp- 60 5)"
mk $OUT/v_late; sed -i '' 's/^100.5\thog_holding/103.5\thog_holding/' $OUT/v_late/e.tsv
chk "hold began after decode -> INVALID" "INVALID" "$(bash $D/verdict.sh $OUT/v_late/h.log $OUT/v_late/e.tsv $OUT/v_late/s.log $OUT/v_late/gate-resp- 60 5)"

echo; echo "RESULT: $PASSN passed, $FAILN failed"; [ $FAILN -eq 0 ]
