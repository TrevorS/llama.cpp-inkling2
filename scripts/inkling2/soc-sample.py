# sample Apple SoC counters through IOReport (no sudo) while a command runs:
# GPU performance state residency, fabric (AFR) state residency, DRAM bytes the GPU reads and writes, GPU loss reasons
# usage: python3 soc-sample.py [--tail <s>] [--interval <ms>] [--states] -- <command...>
#   --tail <s>  summarize only the last s seconds (e.g. the decode after a long prefill)
#   --states    print the residency of every state, not only the summary
import argparse, ctypes as C, subprocess, sys, threading, time

ap = argparse.ArgumentParser()
ap.add_argument("--tail", type=float, default=0)
ap.add_argument("--interval", type=float, default=100)
ap.add_argument("--states", action="store_true")
ap.add_argument("cmd", nargs=argparse.REMAINDER)
a = ap.parse_args()
cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd

cf = C.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
ior = C.CDLL("/usr/lib/libIOReport.dylib")
vp, i64, i32, u64 = C.c_void_p, C.c_int64, C.c_int32, C.c_uint64
for f, res, args in [
    (cf.CFStringCreateWithCString, vp, [vp, C.c_char_p, C.c_uint32]),
    (cf.CFStringGetCString, C.c_bool, [vp, C.c_char_p, C.c_long, C.c_uint32]),
    (cf.CFArrayGetCount, C.c_long, [vp]),
    (cf.CFArrayGetValueAtIndex, vp, [vp, C.c_long]),
    (cf.CFDictionaryGetValue, vp, [vp, vp]),
    (cf.CFDictionaryCreateMutableCopy, vp, [vp, C.c_long, vp]),
    (cf.CFRelease, None, [vp]),
    (ior.IOReportCopyChannelsInGroup, vp, [vp, vp, u64, u64, u64]),
    (ior.IOReportMergeChannels, None, [vp, vp, vp]),
    (ior.IOReportCreateSubscription, vp, [vp, vp, C.POINTER(vp), u64, vp]),
    (ior.IOReportCreateSamples, vp, [vp, vp, vp]),
    (ior.IOReportCreateSamplesDelta, vp, [vp, vp, vp]),
    (ior.IOReportChannelGetGroup, vp, [vp]),
    (ior.IOReportChannelGetSubGroup, vp, [vp]),
    (ior.IOReportChannelGetChannelName, vp, [vp]),
    (ior.IOReportChannelGetFormat, i32, [vp]),
    (ior.IOReportSimpleGetIntegerValue, i64, [vp, i32]),
    (ior.IOReportStateGetCount, i32, [vp]),
    (ior.IOReportStateGetNameForIndex, vp, [vp, i32]),
    (ior.IOReportStateGetResidency, i64, [vp, i32]),
]:
    f.restype, f.argtypes = res, args

def cfs(s): return cf.CFStringCreateWithCString(None, s.encode(), 0x08000100)
def pys(r):
    if not r: return ""
    b = C.create_string_buffer(256)
    cf.CFStringGetCString(r, b, 256, 0x08000100)
    return b.value.decode()

groups = [("GPU Stats", None), ("PMP", "DCS BW"), ("PMP", "AFR BW")]
chans = None
for g, sg in groups:
    c = ior.IOReportCopyChannelsInGroup(cfs(g), cfs(sg) if sg else None, 0, 0, 0)
    if chans is None:
        chans = cf.CFDictionaryCreateMutableCopy(None, 0, c)
    else:
        ior.IOReportMergeChannels(chans, c, None)
subbed = vp()
sub = ior.IOReportCreateSubscription(None, chans, C.byref(subbed), 0, None)
if not sub:
    sys.exit("IOReport subscription failed")
k_chans = cfs("IOReportChannels")

def read(delta):
    out = {}
    arr = cf.CFDictionaryGetValue(delta, k_chans)
    for i in range(cf.CFArrayGetCount(arr)):
        it = cf.CFArrayGetValueAtIndex(arr, i)
        key = (pys(ior.IOReportChannelGetGroup(it)), pys(ior.IOReportChannelGetSubGroup(it)), pys(ior.IOReportChannelGetChannelName(it)))
        fmt = ior.IOReportChannelGetFormat(it)
        if fmt == 1:
            out[key] = ior.IOReportSimpleGetIntegerValue(it, 0)
        elif fmt == 2:
            out[key] = {pys(ior.IOReportStateGetNameForIndex(it, s)): ior.IOReportStateGetResidency(it, s) for s in range(ior.IOReportStateGetCount(it))}
    return out

samples = []  # (t_end, dt, data)
stop = threading.Event()
def loop():
    prev, tp = ior.IOReportCreateSamples(sub, subbed, None), time.time()
    while not stop.is_set():
        time.sleep(a.interval / 1000)
        cur, tc = ior.IOReportCreateSamples(sub, subbed, None), time.time()
        d = ior.IOReportCreateSamplesDelta(prev, cur, None)
        samples.append((tc, tc - tp, read(d)))
        cf.CFRelease(d); cf.CFRelease(prev)
        prev, tp = cur, tc
th = threading.Thread(target=loop); th.start()
t0 = time.time()
rc = subprocess.call(cmd) if cmd else time.sleep(3)
stop.set(); th.join()

sel = [s for s in samples if s[0] - s[1] >= t0 - 0.05]
if a.tail > 0 and sel:
    sel = [s for s in sel if s[0] >= sel[-1][0] - a.tail]
T = sum(s[1] for s in sel)

def state_sum(key):
    tot = {}
    for _, _, d in sel:
        for n, v in (d.get(key) or {}).items():
            tot[n] = tot.get(n, 0) + v
    return tot

def busy_share(tot, idle=("OFF", "IDLE", "DOWN")):
    on = {n: v for n, v in tot.items() if n not in idle}
    s_all, s_on = sum(tot.values()) or 1, sum(on.values()) or 1
    return on, s_on / s_all

def counter(g, sg, name):
    return sum(d.get((g, sg, name), 0) for _, _, d in sel)

print("window %.1f s" % T)
for key, label in [(("GPU Stats", "GPU Performance States", "GPUPH"), "GPU P-state"), (("GPU Stats", "AFR Performance States", "AFRSTATE"), "fabric AFR state")]:
    tot = state_sum(key)
    if not tot:
        continue
    on, share = busy_share(tot)
    s_on = sum(on.values()) or 1
    top = sorted(on.items(), key=lambda kv: -kv[1])[:3]
    print("%-18s active %5.1f%%, of active time: %s" % (label, 100 * share, "  ".join("%s %.0f%%" % (n, 100 * v / s_on) for n, v in top if v)))
    if a.states:
        print("   " + "  ".join("%s %.1f%%" % (n, 100 * v / (sum(tot.values()) or 1)) for n, v in tot.items() if v))
# DRAM traffic of all agents, as a histogram of short samples (bins of 32 GB/s)
for sg in ("DCS BW",):
    tot = state_sum(("PMP", sg, "AMCC RD+WR"))
    if tot:
        bins = sorted((float(n.strip().rstrip("GB/s")), v) for n, v in tot.items())
        n = sum(v for _, v in bins) or 1
        mean = sum((b - 16) * v for b, v in bins) / n
        def pct(q):
            acc = 0
            for b, v in bins:
                acc += v
                if acc >= q * n: return b
        hi = sum(v for b, v in bins if b > 384) / n
        print("%-18s DRAM read+write mean %.0f GB/s, p50 %.0f, p90 %.0f, %.0f%% of samples above 384 GB/s" % ("memory (" + sg + ")", mean, pct(0.5), pct(0.9), 100 * hi))
        if a.states:
            print("   " + "  ".join("%.0f:%.0f%%" % (b, 100 * v / n) for b, v in bins if v))
for key, label in [(("GPU Stats", "PMU Loop Lost Performance Reason Code States", "PMU_RC"), "GPU loss reasons"), (("GPU Stats", "CLTM-induced GPU Performance States", "GPU_CLTM"), "thermal (CLTM)"), (("GPU Stats", "GPU Software Performance States", "GPU_SW"), "GPU SW states")]:
    tot = state_sum(key)
    s = sum(tot.values()) or 1
    top = sorted(tot.items(), key=lambda kv: -kv[1])[:4]
    print("%-18s %s" % (label, "  ".join("%s %.0f%%" % (n, 100 * v / s) for n, v in top if v)))
sys.exit(rc)
