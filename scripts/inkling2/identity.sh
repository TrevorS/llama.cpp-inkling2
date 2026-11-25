#!/bin/zsh
# speculation on vs off through the server, temperature 0.8 with a fixed seed: tokens and log-probabilities must be the same as spec off
# sampling shows a logit difference within a few tokens, greedy text can hide it for a long time
# cases, run in this order on one server per setting:
#   keep  - replay.py, 3 turns, the reasoning is sent back (the prompt cache is reused as it is; crosses the SWA ring end)
#   strip - replay.py, 4 turns, the reasoning is not sent back (a checkpoint restore each turn)
#   end   - replay.py, 4 turns, max_tokens 1500 (the replies end by themselves)
#   edit  - edit.py (text only)
# usage: identity.sh [settings, default: k2 k7 ng]; kN = DSpark with N draft tokens,
# ng = n-gram + DSpark at the default draft length of serve.sh; env CASES, PORT, and DRAFT (the draft file, see serve.sh)
S=${0:A:h}
D=${INKLING_DIR:-$HOME/models/Inkling-2-Mini-GGUF}/bench
W=$D/identity
PORT=${PORT:-18093}
mkdir -p $W
cases=(${=CASES:-keep strip end edit})
cfgs=($@)
(( $#cfgs )) || cfgs=(k2 k7 ng)

run() {
  local cfg=$1; shift
  env "$@" LLAMA_PORT=$PORT $S/serve.sh > $W/serve-$cfg.log 2>&1 &
  local pid=$!
  for i in {1..240}; do curl -s 127.0.0.1:$PORT/health 2>/dev/null | grep -q ok && break; sleep 1; done
  for c in $cases; do
    case $c in
      keep)  LOGPROBS=1 TURNS=3 python3 $S/replay.py $PORT $W/keep-$cfg.json 20000 0.8 > /dev/null;;
      strip) LOGPROBS=1 STRIP=1 python3 $S/replay.py $PORT $W/strip-$cfg.json 20000 0.8 > /dev/null;;
      end)   LOGPROBS=1 MAXTOK=1500 python3 $S/replay.py $PORT $W/end-$cfg.json 20000 0.8 > /dev/null;;
      edit)  python3 $S/edit.py $PORT $W/edit-$cfg.json 0.8 > /dev/null;;
    esac
  done
  kill $pid; wait $pid 2>/dev/null
}

run s0 SPEC=0
bad=0
for cfg in $cfgs; do
  case $cfg in
    k*)  envs=(NGRAM=0 SPEC_N=${cfg#k});;
    ng)  envs=(NGRAM=1);;
    *)   print "unknown setting $cfg"; exit 1;;
  esac
  run $cfg $envs
  for c in $cases; do
    print "== $cfg $c"
    python3 $S/identity.py $W/$c-s0.json $W/$c-$cfg.json || bad=1
  done
done
(( bad )) && print "identity: DIFFERENCES found" || print "identity: all same"
exit $bad
