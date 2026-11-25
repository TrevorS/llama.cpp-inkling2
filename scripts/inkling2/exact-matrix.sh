#!/bin/zsh
# wide check of rule 1: for 3 prompts, decode in batches of 1..8 with 7 rollback slots and in batches of 1 and 3 with 2 slots;
# every run must give the logits of batch 1, bitwise. Then rejected drafts across the end of the SWA cache ring, at 2 depths.
# Then the dispatch census against the expected count.
# usage: exact-matrix.sh [expected dispatches per decode step, default 2105]
S=${0:A:h}
L=${S:h:h}
D=${INKLING_DIR:-$HOME/models/Inkling-2-Mini-GGUF}/bench
M=${D:h}/Inkling-2-Mini-UD-Q8_K_XL.upstream.gguf
B=$L/build/bin
W=$D/exact-matrix
mkdir -p $W

if [[ ! -x $D/decode-dump || $B/libllama.dylib -nt $D/decode-dump ]]; then
  c++ -std=c++17 -O2 -Wno-deprecated-declarations -I$L/include -I$L/ggml/include $S/decode-dump.cpp \
    -L$B -lllama -lggml -lggml-base -Wl,-rpath,$B -o $D/decode-dump || exit 1
fi

prompts=(
  ""
  "$(head -c 600 $S/par.py)"
  "$(head -c 600 $S/gpu-clocks.md)"
)
bad=0
for p in 1 2 3; do
  args=($M $W/ref.bin 48)
  [[ -n $prompts[$p] ]] && args+=("$prompts[$p]")
  BATCH=1 N_RS=7 $D/decode-dump $args 2>/dev/null
  for cfg in 2:7 3:7 4:7 5:7 6:7 7:7 8:7 1:2 3:2; do
    b=${cfg%%:*}; r=${cfg##*:}
    args[2]=$W/out.bin
    BATCH=$b N_RS=$r $D/decode-dump $args 2>/dev/null
    cmp -s $W/ref.bin $W/out.bin || { print "prompt $p BATCH=$b N_RS=$r: DIFFERS"; bad=$((bad + 1)); }
  done
  print "prompt $p: 9 settings compared with batch 1"
done
print "bitwise: $((27 - bad))/27 same"

# rejected drafts: each batch keeps 1 token (KEEP) or a random count (ROLL), the other tokens are wrong ones and are removed
# the 120 tokens cross the end of the ring: position 1536 for one slot, 9216 (2 x 4608) for the server layout
bad=0
n=0
for lay in "1480:" "9150:1"; do
  d=${lay%%:*}
  args=($M $W/ref.bin 120 @$L/common/speculative.cpp)
  DEPTH=$d SERVER=${lay##*:} BATCH=1 N_RS=7 $D/decode-dump $args 2>/dev/null
  args[2]=$W/out.bin
  for cfg in "8 KEEP=1" "3 KEEP=1" "8 ROLL=1" "5 ROLL=2"; do
    env DEPTH=$d SERVER=${lay##*:} BATCH=${cfg%% *} N_RS=7 ${cfg##* } $D/decode-dump $args 2>/dev/null
    cmp -s $W/ref.bin $W/out.bin || { print "depth $d BATCH=$cfg: DIFFERS"; bad=$((bad + 1)); }
    n=$((n + 1))
  done
done
print "rejected drafts across the ring end: $((n - bad))/$n same"

GGML_METAL_GRAPH_DEBUG=1 GGML_METAL_N_CB=0 N_RS=2 $D/decode-dump $M $W/out.bin 3 2>$W/census.log
python3 $S/census.py $W/census.log --check ${1:-2105} | sed -n '1p;$p'
