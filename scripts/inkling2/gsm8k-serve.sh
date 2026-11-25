#!/bin/zsh
# start serve.sh on port 18097 (32k context) with the given env, run gsm8k.py on it, stop it
# usage: gsm8k-serve.sh <gsm8k.jsonl> <out.jsonl> <n|all> [ENV=val ...]; SERVE=<path> uses the serve.sh of another checkout
# stop the live server first: two copies of the model do not fit in 64 GB, and the second one fails with Metal out-of-memory errors
S=${0:A:h}
data=$1; out=$2; n=$3; shift 3
env "$@" CTX=32768 LLAMA_PORT=18097 ${SERVE:-$S/serve.sh} > ${out%.jsonl}-server.log 2>&1 &
pid=$!
for i in {1..240}; do curl -s 127.0.0.1:18097/health 2>/dev/null | grep -q ok && break; sleep 1; done
if [[ $n == all ]]; then python3 $S/gsm8k.py 18097 $data $out; else python3 $S/gsm8k.py 18097 $data $out $n; fi
kill $pid; wait $pid 2>/dev/null
