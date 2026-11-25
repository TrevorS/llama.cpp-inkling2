# per-token wall and GPU time from the XGPU lines of gpu-timeline.patch (apply it, build, run with X_GPUT=1 2>log)
# usage: python3 gpu-timeline.py <log> [--gaps]
import re, sys, statistics as st, collections

ev = []
for line in open(sys.argv[1], errors="ignore"):
    m = re.search(r"XGPU cb (\d+) nodes (\d+) enc_end ([0-9.]+) gpu ([0-9.]+)-([0-9.]+)", line)
    if m:
        ev.append((int(m.group(1)), float(m.group(3)), float(m.group(4)), float(m.group(5))))

# the main command buffer has the highest index and starts each graph
main = max(e[0] for e in ev)
starts = [e[2] for e in ev if e[0] == main]
wall = [b - a for a, b in zip(starts, starts[1:])][4:]
gpu, cur = [], 0.0
for e in ev:
    if e[0] == main and cur:
        gpu.append(cur)
        cur = 0.0
    cur += e[3] - e[2]
gpu = gpu[4:]
print("command buffers %d  wall med %.3f mean %.3f ms  gpu med %.3f ms" % (main + 1, st.median(wall), st.mean(wall), st.median(gpu)))

if "--gaps" in sys.argv:
    ev.sort(key=lambda e: e[2])
    gaps = collections.defaultdict(list)
    for a, b in zip(ev, ev[1:]):
        gaps[(a[0], b[0])].append(b[2] - a[3])
    for k in sorted(gaps):
        print("  idle cb %d -> cb %d: %.3f ms" % (k[0], k[1], st.median(gaps[k])))
