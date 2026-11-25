#!/bin/zsh
# speculation sweep through the server: each setting runs short.py, edit.py and replay.py (3 turns, 20000 chars), greedy,
# with the GPU at 50 C or below before each prompt set; every setting's text must equal spec off (s0, run first)
# settings: s0 = spec off, kN = DSpark with N draft tokens, ngN = n-gram drafts + DSpark with N draft tokens
# stops a live server on port 8080 for the memory and starts it again at the end
# usage: spec-sweep.sh <out dir> [settings, default: s0 k2 k3 k5 k7 ng2 ng3 ng5 ng7]
S=${0:A:h}
M=${INKLING_DIR:-$HOME/models/Inkling-2-Mini-GGUF}
W=${1:?out dir}; shift
mkdir -p $W
cfgs=($@)
(( $#cfgs )) || cfgs=(s0 k2 k3 k5 k7 ng2 ng3 ng5 ng7)
[[ $cfgs[1] == s0 ]] || cfgs=(s0 $cfgs)

cool() {
  local t
  sleep 20
  for i in {1..40}; do
    t=$(mactop --headless --count 1 -i 250 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d; print("%.0f" % d["soc_metrics"]["gpu_temp"])' 2>/dev/null)
    (( ${t:-99} > 0 && ${t:-99} <= 50 )) && break
    sleep 10
  done
  print -n "[${t} C] "
}

live=0; pgrep -f "llama-server.*port 8080" > /dev/null && live=1
pkill -f "llama-server.*port 8080"; while pgrep -f "llama-server.*port 8080" > /dev/null; do sleep 1; done

for cfg in $cfgs; do
  case $cfg in
    s0)  envs=(SPEC=0);;
    k*)  envs=(NGRAM=0 SPEC_N=${cfg#k});;
    ng*) envs=(NGRAM=1 SPEC_N=${cfg#ng});;
    *)   print "unknown setting $cfg"; exit 1;;
  esac
  env $envs LLAMA_PORT=18092 $S/serve.sh > $W/serve-$cfg.log 2>&1 &
  pid=$!
  for i in {1..240}; do curl -s 127.0.0.1:18092/health 2>/dev/null | grep -q ok && break; sleep 1; done
  print "== $cfg ($envs)"
  cool; python3 $S/short.py 18092 $W/short-$cfg.json
  cool; python3 $S/edit.py 18092 $W/edit-$cfg.json
  cool; TURNS=3 python3 $S/replay.py 18092 $W/replay-$cfg.json 20000
  kill $pid; wait $pid 2>/dev/null
done

print "== same text as s0"
for cfg in ${cfgs:1}; do
  for set in short edit; do
    cmp -s $W/$set-s0.json $W/$set-$cfg.json && print -n "$cfg $set same, " || print -n "$cfg $set DIFF, "
  done
  python3 -c 'import json,sys; a,b=[[t["msg"] for t in json.load(open(f))] for f in sys.argv[1:3]]; print("replay", "same" if a==b else "DIFF")' $W/replay-s0.json $W/replay-$cfg.json
done

(( live )) || exit 0
cd $M && nohup ${S}/serve.sh > server.log 2>&1 &
for i in {1..240}; do curl -s 127.0.0.1:8080/health 2>/dev/null | grep -q ok && break; sleep 1; done
print "live server: $(curl -s 127.0.0.1:8080/health)"
