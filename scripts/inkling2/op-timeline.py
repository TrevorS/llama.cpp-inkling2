# per-op GPU time and bytes of one graph, from the X_OPT probe (apply op-timeline.patch and build first)
# usage: GGML_METAL_N_CB=2 X_OPT=1 llama-simple -m <model> -n 16 "<prompt>" 2> op.log; python3 op-timeline.py op.log [--tokens 1] [--peak 546] [--top 40]
# apply: git apply scripts/inkling2/op-timeline.patch; undo: git apply -R scripts/inkling2/op-timeline.patch
# GGML_METAL_N_CB=2: the probe hooks the multi-buffer path (the default chunked path opens more sample buffers than Metal allows)
# each op runs in its own encoder, so the times are per op but the probe adds a gap of a few us after each op
# the GPU lowers its clock when it idles in these gaps, more so at short depth: compare runs at the same depth (llama-bench -d 32768 keeps it busier) and read the times as shares, not absolute cost
# "busy" counts time where at least one op runs; an op is charged only for the time after the previous op ended
import argparse, collections, re

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--tokens", type=int, default=1, help="use graphs with this many tokens (from the first RMS_NORM)")
ap.add_argument("--peak", type=float, default=546.0, help="memory bandwidth in GB/s for the floor column")
ap.add_argument("--top", type=int, default=40)
a = ap.parse_args()

graphs = collections.defaultdict(list)
for line in open(a.log, errors="replace"):
    if not line.startswith("XOP\t"):
        continue
    _, g, cb, idx, n, op, name, ne0, ne1, b, t0, t1 = line.rstrip("\n").split("\t")
    graphs[int(g)].append(dict(op=op, name=name, n=int(n), ne0=int(ne0), ne1=int(ne1), bytes=float(b), t0=int(t0), t1=int(t1)))

def n_tokens(recs):
    for r in recs:
        if r["op"] == "RMS_NORM":
            return r["ne1"]
    return -1

sel = [g for g in sorted(graphs) if n_tokens(graphs[g]) == a.tokens]
if len(sel) > 2:
    sel = sel[1:]  # first one warms up
if not sel:
    raise SystemExit("no graphs with %d tokens" % a.tokens)

def key(r):
    stem = re.sub(r"\s*\(.*$", "", r["name"])
    stem = re.sub(r"[-_]?l?\d+$", "", stem)
    if stem in ("node", ""):
        stem = "[%d,%d]" % (r["ne0"], r["ne1"])
    if r["op"] in ("MUL_MAT", "MUL_MAT_ID"):
        stem += " %.1fMB" % (r["bytes"] / 1e6)
    return "%s%s %s" % (r["op"], "+%d" % (r["n"] - 1) if r["n"] > 1 else "", stem)

agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0])  # count, excl ns, raw ns, bytes
busy = span = raw = 0.0
for g in sel:
    recs = sorted(graphs[g], key=lambda r: r["t0"])
    end = recs[0]["t0"]
    for r in recs:
        excl = max(0, r["t1"] - max(r["t0"], end))
        end = max(end, r["t1"])
        k = agg[key(r)]
        k[0] += 1; k[1] += excl; k[2] += r["t1"] - r["t0"]; k[3] += r["bytes"]
        busy += excl; raw += r["t1"] - r["t0"]
    span += max(r["t1"] for r in recs) - recs[0]["t0"]

ng = len(sel)
tot_b = sum(v[3] for v in agg.values()) / ng
floor = lambda b: b / (a.peak * 1e3)  # bytes -> us
print("graphs %d (tokens %d), ops/graph %d" % (ng, a.tokens, sum(v[0] for v in agg.values()) / ng))
print("per graph: GPU span %.0f us, busy %.0f us, sum of op times %.0f us" % (span / ng / 1e3, busy / ng / 1e3, raw / ng / 1e3))
print("per graph: %.1f MB moved, floor at %.0f GB/s %.0f us, busy at %.0f GB/s" % (tot_b / 1e6, a.peak, floor(tot_b), tot_b / (busy / ng)))
print()
print("%-44s %6s %9s %6s %9s %7s %9s %9s" % ("op (+fused) name/shape", "n", "us", "%", "MB", "GB/s", "floor us", "excess"))
rows = sorted(agg.items(), key=lambda kv: -kv[1][1])
for k, (c, e, _, b) in rows[:a.top]:
    us = e / ng / 1e3; mb = b / ng / 1e6
    print("%-44s %6.0f %9.1f %6.1f %9.2f %7.0f %9.1f %9.1f" % (k[:44], c / ng, us, 100 * e / busy, mb, (b / e) if e else 0, floor(b / ng), us - floor(b / ng)))
rest = rows[a.top:]
if rest:
    e = sum(v[1] for _, v in rest); b = sum(v[3] for _, v in rest); c = sum(v[0] for _, v in rest)
    print("%-44s %6.0f %9.1f %6.1f %9.2f %7s %9.1f %9.1f" % ("(other %d keys)" % len(rest), c / ng, e / ng / 1e3, 100 * e / busy, b / ng / 1e6, "", floor(b / ng), e / ng / 1e3 - floor(b / ng)))

print()
print("by bytes per op:")
buckets = [("< 64 KB", 0, 64e3), ("64 KB - 1 MB", 64e3, 1e6), ("1 - 16 MB", 1e6, 16e6), (">= 16 MB", 16e6, 1e18)]
print("%-14s %6s %9s %6s %9s %9s" % ("", "n", "us", "%", "floor us", "excess"))
for lab, lo, hi in buckets:
    c = e = b = 0.0
    for g in sel:
        recs = sorted(graphs[g], key=lambda r: r["t0"])
        end = recs[0]["t0"]
        for r in recs:
            ex = max(0, r["t1"] - max(r["t0"], end)); end = max(end, r["t1"])
            if lo <= r["bytes"] < hi:
                c += 1; e += ex; b += r["bytes"]
    print("%-14s %6.0f %9.1f %6.1f %9.1f %9.1f" % (lab, c / ng, e / ng / 1e3, 100 * e / busy, floor(b / ng), e / ng / 1e3 - floor(b / ng)))
