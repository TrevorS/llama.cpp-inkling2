#!/bin/zsh
# thermal check (see gpu-clocks.md): decode from cool and right after a long prefill, decode at 32k depth three times in a row,
# then a long prefill alone; each with GPU P-states, thermal caps (CLTM) and DRAM traffic from soc-sample.py, plus temperature and fans
# needs mactop (brew install mactop) for temperatures and fan speed; stops the live server on port 8080 and starts it again at the end
# usage: heat.sh [build/bin dir, default this checkout]
S=${0:A:h}; B=${1:-${S:h:h}/build/bin}
M=${INKLING_DIR:-$HOME/models/Inkling-2-Mini-GGUF}
G=$M/Inkling-2-Mini-UD-Q8_K_XL.upstream.gguf
P=(python3 $S/soc-sample.py)
temps() {
  mactop --headless --count 1 -i 250 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d; print("   gpu %.0f C" % d["soc_metrics"]["gpu_temp"], end=", ")'
  mactop --fan-status 2>/dev/null | python3 -c 'import json,sys; print("fans rpm", [f["rpm"] for f in json.load(sys.stdin)])'
}
show() { grep -E "(tg|pp)[0-9]+|^GPU P|^fabric|^thermal|^memory"; }

pkill -f "llama-server.*port 8080"; while pgrep -f "llama-server.*port 8080" > /dev/null; do sleep 1; done

echo "=== decode from cool (idle 120 s)"; sleep 120; temps
$P --tail 5 -- $B/llama-bench -m $G -fa 1 -p 0 -n 512 -r 1 2>/dev/null | show; temps
echo "=== decode right after 45 s of prefill"
$B/llama-bench -m $G -fa 1 -p 2048 -n 0 -r 12 2>/dev/null | show; temps
$P --tail 5 -- $B/llama-bench -m $G -fa 1 -p 0 -n 512 -r 1 2>/dev/null | show; temps
for i in 1 2 3; do
  echo "=== decode at 32k depth, run $i (counters cover the last 3 s, the decode)"
  $P --tail 3 -- $B/llama-bench -m $G -fa 1 -p 0 -n 128 -d 32768 -r 1 2>/dev/null | show; temps
done
echo "=== prefill pp2048 x12"
$P -- $B/llama-bench -m $G -fa 1 -p 2048 -n 0 -r 12 2>/dev/null | show; temps

cd $M && nohup ./serve.sh > server.log 2>&1 &
for i in {1..240}; do curl -s 127.0.0.1:8080/health 2>/dev/null | grep -q ok && break; sleep 1; done
echo "live server: $(curl -s 127.0.0.1:8080/health)"
