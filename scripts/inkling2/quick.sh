#!/bin/zsh
# quick checks for the Inkling 2 llama.cpp fork; reference data and outputs live in $INKLING_DIR/bench
# usage: quick.sh [--base <build/bin dir>] [--agent] [--depth] [--ops <OP>]
#   default: spec exactness, greedy reference, short perplexity, pp512/tg128 at depth 0
#   --base   interleave the speed bench with another build: base, new, base, new
#   --agent  3 agent prompts through the server with spec on (tok/s, draft acceptance)
#   --depth  pp512/tg64 at 32k context
#   --ops    test-backend-ops for one op on Metal, e.g. --ops FLASH_ATTN_EXT
S=${0:A:h}
L=${S:h:h}
D=${INKLING_DIR:-$HOME/models/Inkling-2-Mini-GGUF}/bench
M=${D:h}/Inkling-2-Mini-UD-Q8_K_XL.upstream.gguf
B=$L/build/bin
zparseopts -D -E -base:=base_opt -agent=agent -depth=depth -ops:=ops_opt
BASE=${base_opt[2]}
OPS=${ops_opt[2]}
T0=$SECONDS

step() { T1=$SECONDS; print -n "$1: "; }
took() { print "  (${$(( SECONDS - T1 ))}s)"; }

# spec exactness: K tokens one by one vs as one batch, logits must match bitwise
if [[ ! -x $D/batch-exact || $B/libllama.dylib -nt $D/batch-exact ]]; then
  c++ -std=c++17 -O2 -Wno-deprecated-declarations -I$L/include -I$L/ggml/include $S/batch-exact.cpp \
    -L$B -lllama -lggml -lggml-base -Wl,-rpath,$B -o $D/batch-exact || exit 1
fi
step "exact"
P="$(cat $L/docs/build.md)"
out=$( { $D/batch-exact $M 3 0; NTOK=2050 $D/batch-exact $M 4 0 "$P"; } 2>/dev/null | grep "^token")
bad=$(print -r -- "$out" | grep -vc "logits same")
[[ $bad == 0 ]] && print -n "ok ($(print -r -- "$out" | wc -l | tr -d ' ') tokens bitwise)" || print -n "FAIL ($bad tokens differ)"
took

# greedy reference: 64 tokens, byte compare
step "greedy"
$B/llama-simple -m $M -n 64 "Write a haiku about compilers, then explain the key idea of SSA form in two sentences." 2>/dev/null > $D/out-simple.txt
cmp -s $D/out-simple.txt $D/ref-simple.txt && print -n "same as reference" || print -n "CHANGED, see $D/out-simple.txt"
took

# accuracy: perplexity on 4 chunks of the docs text, the first run sets the reference
step "ppl"
ppl=$($B/llama-perplexity -m $M -f $D/eval.txt -c 2048 -fa on --chunks 4 2>&1 | sed -n 's/.*Final estimate: PPL = \([0-9.]*\).*/\1/p')
[[ -f $D/ref-ppl.txt ]] || print $ppl > $D/ref-ppl.txt
print -n "$ppl (reference $(cat $D/ref-ppl.txt))"
took

bench() { # <bin dir> <label> <llama-bench args...>
  local bin=$1 label=$2; shift 2
  print "  $label $($bin/llama-bench -m $M -fa 1 "$@" -o csv 2>/dev/null | tail -n +2 | tr -d '"' | awk -F, '{printf "%.1f ", $(NF-1)}')"
}

# speed at depth 0, tok/s for pp512 then tg128
step "speed d0 (pp512 tg128)"; print
if [[ -n $BASE ]]; then
  for r in 1 2; do bench $BASE base -p 512 -n 128 -r 2; bench $B new -p 512 -n 128 -r 2; done
else
  bench $B new -p 512 -n 128 -r 2
fi
print -n " "; took

if [[ -n $depth ]]; then
  step "speed d32k (pp512 tg64)"; print
  [[ -n $BASE ]] && bench $BASE base -p 512 -n 64 -d 32768 -r 1
  bench $B new -p 512 -n 64 -d 32768 -r 1
  print -n " "; took
fi

if [[ -n $agent ]]; then
  step "agent (spec on)"
  CTX=16384 LLAMA_PORT=18095 $S/serve.sh > $D/agent-server.log 2>&1 &
  pid=$!
  for i in {1..120}; do curl -s 127.0.0.1:18095/health 2>/dev/null | grep -q ok && break; sleep 1; done
  python3 - <<'EOF'
import json, urllib.request
prompts = [
    "Write a Python function that parses an ISO 8601 duration string like P3DT4H5M into total seconds. Include type hints.",
    "Write a bash script that finds the 10 largest files under a directory, skipping .git.",
    "Explain how a B-tree insert splits nodes, step by step, in plain prose.",
]
tok = secs = acc = drafted = 0
for p in prompts:
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": 256, "temperature": 0, "reasoning_effort": "low"}
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18095/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})))
    t = r["timings"]
    tok += t["predicted_n"]; secs += t["predicted_ms"]/1000
    acc += t.get("draft_n_accepted", 0); drafted += t.get("draft_n", 0)
print("%.1f tok/s, draft %d/%d" % (tok/secs, acc, drafted), end="")
EOF
  kill $pid; wait $pid 2>/dev/null
  took
fi

if [[ -n $OPS ]]; then
  step "ops $OPS"
  print -n "$($B/test-backend-ops -o $OPS -b MTL0 2>&1 | grep -E 'tests passed' | tr -s ' ')"
  took
fi

print "total ${$(( SECONDS - T0 ))}s"
