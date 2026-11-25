#!/bin/zsh
# start serve.sh with given env/args on port 18095, send N concurrent prompts, stop it
# usage: par.sh <N> <ENV=val ...> -- <server args...>
S=${0:A:h}; D=${INKLING_DIR:-$HOME/models/Inkling-2-Mini-GGUF}/bench; N=$1; shift
envs=(); while [[ $# -gt 0 && $1 != -- ]]; do envs+=$1; shift; done; shift
env $envs CTX=${CTX:-32768} LLAMA_PORT=18095 $S/serve.sh "$@" > $D/par-server.log 2>&1 &
pid=$!
for i in {1..180}; do curl -s 127.0.0.1:18095/health 2>/dev/null | grep -q ok && break; sleep 1; done
python3 $S/par.py 18095 1 > /dev/null  # warm up
print -n "$envs $@: "; python3 $S/par.py 18095 $N $D/par-out-${N}-${(j:_:)envs}.json
kill $pid; wait $pid 2>/dev/null
