#!/bin/zsh
# benchmark a list of commits under the same conditions: each is built in a separate worktree, then pp512/tg128 at depth 0, 8k and 32k
# (llama-bench, 1 run each, 60 s cool-down before each commit); appends to the csv, plot it with history-plot.py
# stops the live server on port 8080 for the memory and starts it again at the end
# usage: history.sh <worktree dir> <out.csv> <commit...>
#   the worktree is created on first use (git worktree add --detach) and configured like build/; its build/ is reused between commits
S=${0:A:h}; L=${S:h:h}
B=$1; out=$2; shift 2
M=${INKLING_DIR:-$HOME/models/Inkling-2-Mini-GGUF}
G=$M/Inkling-2-Mini-UD-Q8_K_XL.upstream.gguf

if [[ ! -d $B ]]; then
  git -C $L worktree add --detach $B HEAD || exit 1
  cmake -S $B -B $B/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_METAL_EMBED_LIBRARY=ON > /dev/null || exit 1
fi

pkill -f "llama-server.*port 8080"; while pgrep -f "llama-server.*port 8080" > /dev/null; do sleep 1; done

for c in "$@"; do
  git -C $B checkout -q -- . && git -C $B switch -q --detach $c || { echo "$c: checkout failed"; continue; }
  cmake --build $B/build -j --target llama-bench > /dev/null 2>&1 || { echo "$c: build failed"; continue; }
  sleep 60
  $B/build/bin/llama-bench -m $G -fa 1 -p 512 -n 128 -d 0,8192,32768 -r 1 -o csv 2> /dev/null | tail -n +2 | while read -r line; do
    echo "$c,$(git -C $B log -1 --format=%cI $c),${line//$HOME/~}" >> $out
  done
  echo "$c: $(grep -c "^$c," $out) rows"
done

cd $M && nohup ./serve.sh > server.log 2>&1 &
for i in {1..240}; do curl -s 127.0.0.1:8080/health 2>/dev/null | grep -q ok && break; sleep 1; done
echo "live server: $(curl -s 127.0.0.1:8080/health)"
