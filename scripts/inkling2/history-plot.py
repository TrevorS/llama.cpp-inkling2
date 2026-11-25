# plot history.sh results: two panels (pp512, tg128) over the commit dates, one line per depth
# usage: uvx --with matplotlib python3 history-plot.py <in.csv> <out.png>; edit `marks` to label the commits that matter
import csv, sys
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

src, out = sys.argv[1], sys.argv[2]
pts = {}
for r in csv.reader(open(src)):
    c, date = r[0], datetime.fromisoformat(r[1]).replace(tzinfo=None)
    pp, d, ts = int(r[-8]), int(r[-6]), float(r[-2])
    pts.setdefault(("pp512" if pp else "tg128", d), []).append((date, ts, c))

colors = {0: "#2a78d6", 8192: "#eb6834", 32768: "#1baf7a"}
names = {0: "depth 0", 8192: "depth 8k", 32768: "depth 32k"}
marks = [("2026-11-20T23:17", "first fusions,\nNVFP4 decode"), ("2026-11-21T19:12", "FA rel bias"),
         ("2026-11-22T10:25", "8 cmd buffers,\ndepth mix"), ("2026-11-23T03:26", "SITU + conv"), ("2026-11-23T08:07", "4 more\nfusions")]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "font.family": "Helvetica"})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150, facecolor="#fcfcfb")
for ax, test, title in [(axes[0], "pp512", "Prompt processing, pp512 (tok/s)"), (axes[1], "tg128", "Decode, tg128 (tok/s)")]:
    ax.set_facecolor("#fcfcfb")
    for d in (0, 8192, 32768):
        s = sorted(pts[(test, d)])
        xs, ys = [p[0] for p in s], [p[1] for p in s]
        ax.plot(xs, ys, color=colors[d], lw=2, marker="o", ms=4.5, label=names[d], zorder=3)
        fmt = "%.0f" if test == "pp512" else "%.1f"
        ax.annotate(fmt % ys[0], (xs[0], ys[0]), textcoords="offset points", xytext=(-6, 0), ha="right", va="center", fontsize=9, color="#52514e")
        ax.annotate(fmt % ys[-1], (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, 0), ha="left", va="center", fontsize=9, color="#0b0b0b")
    lo, hi = ax.get_ylim()
    hi = hi + (hi - lo)*0.22  # headroom above the data for the mark labels
    ax.set_ylim(lo, hi)
    for i, (t, lab) in enumerate(marks):
        x = datetime.fromisoformat(t)
        ax.axvline(x, color="#c3c2b7", lw=1, ls=(0, (3, 3)), zorder=1)
        ax.text(x, hi - (hi - lo)*(0.01 + 0.085*(i % 2)), lab, fontsize=7.5, color="#52514e", ha="center", va="top",
                bbox=dict(fc="#fcfcfb", ec="none", pad=1))
    ax.set_title(title, loc="left", fontsize=12, color="#0b0b0b")
    ax.grid(axis="y", color="#e9e8e4", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.margins(x=0.1)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
fig.suptitle("Inkling 2 Mini on M4 Max, Metal: each milestone build benchmarked on 2026-11-23 (llama-bench, 1 run each)",
             x=0.01, ha="left", fontsize=10, color="#52514e")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out, facecolor=fig.get_facecolor())
print(out)
