# count the Metal dispatches of one decode step by op, from the GGML_METAL_GRAPH_DEBUG=1 log of decode-dump
# a fusion that stops matching (after a merge or a graph change) shows up as more dispatches
# usage: GGML_METAL_GRAPH_DEBUG=1 GGML_METAL_N_CB=0 N_RS=2 decode-dump model.gguf /dev/null 3 2>log; python3 census.py log [--check N]
import collections, re, sys
graphs, cur = [], None
for line in open(sys.argv[1], errors="ignore"):
    m = re.search(r"node\[\s*(\d+)\] - (\S+)", line)
    if m:
        if int(m.group(1)) == 0 or cur is None:
            cur = []
            graphs.append(cur)
        cur.append([m.group(2), 1])
        continue
    m = re.search(r"fuse (\d+) ops", line)
    if m and cur:
        cur[-1][1] = int(m.group(1))
g = graphs[-1]
count, nodes = collections.Counter(), collections.Counter()
for op, n in g:
    key = op + (" (fused)" if n > 1 else "")
    count[key] += 1
    nodes[key] += n
print("dispatches per decode step: %d" % len(g))
for key, v in count.most_common():
    print("%5d  %-26s %5d nodes" % (v, key, nodes[key]))
if "--check" in sys.argv:
    want = int(sys.argv[sys.argv.index("--check") + 1])
    if len(g) != want:
        sys.exit("FAIL: %d dispatches, expected %d" % (len(g), want))
    print("ok: matches the expected %d" % want)
