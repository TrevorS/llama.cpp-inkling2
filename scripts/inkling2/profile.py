# estimate the GPU time of one Inkling 2 decode step per op group from test-backend-ops perf (best of 2)
# usage: python3 profile.py   (apply profile-cases.patch to tests/test-backend-ops.cpp and build it first)
import pathlib, re, subprocess, collections
B = str(pathlib.Path(__file__).resolve().parents[2] / "build/bin/test-backend-ops")
runs = [
    ("MUL_MAT", r"type_a=(bf16|f32),type_b=f32,m=(3072|1024|2304|201024|24|64|384|32|128),n=(1|8|16),k=(2304|3072|4608),bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,o=1,"),
    ("MUL_MAT_ID", r"type_a=(nvfp4|bf16),type_b=f32,n_mats=128,n_used=8,b=0,m=2304,n=(1|8|16),k=(2304|1152),"),
    ("FLASH_ATTN_EXT", r"hsk=128,hsv=128,nh=8,nr23=\[3,1\],kv=(1024|32768|131072),nb=(1|8|16),mask=1,sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=f16,type_V=f16,"),
]
t = {}
for _ in range(2):
    for op, pat in runs:
        out = subprocess.run([B, "perf", "-o", op, "-b", "MTL0", "-p", pat], capture_output=True, text=True).stdout
        for line in out.splitlines():
            m = re.search(r"%s\((.*)\):.* ([0-9.]+) us/run" % op, line)
            if not m: continue
            key = (op, m.group(1)); us = float(m.group(2))
            t[key] = min(us, t.get(key, 1e18))
def get(op, **kw):
    for (o, v), us in t.items():
        if o == op and all(re.search(r"(^|,)%s=%s," % (k, x), v + ",") for k, x in kw.items()):
            return us
    raise KeyError((op, kw))
for n in (1, 8, 16):
    g = collections.OrderedDict()
    mm = lambda m, k, ty="bf16": get("MUL_MAT", type_a=ty, m=m, n=n, k=k)
    g["attn proj q+gate (3072)"] = 96 * mm(3072, 2304)
    g["attn proj k+v (1024)"]    = 96 * mm(1024, 2304)
    g["attn out (2304, k3072)"]  = 48 * mm(2304, 3072)
    g["rel+sink (384, 24)"]      = 48 * (mm(384, 2304) + mm(24, 2304))
    g["loop ctrl down (64, k4608)"] = 96 * mm(64, 4608)
    g["router f32 (128)"]        = 48 * mm(128, 2304, "f32")
    g["conv proj (32)"]          = 8 * mm(32, 2304)
    g["output head"]             = mm(201024, 2304)
    me = lambda ty, k: get("MUL_MAT_ID", type_a=ty, n=n, k=k)
    g["experts nvfp4 (44 runs)"] = 44 * (me("nvfp4", 2304) + me("nvfp4", 1152))
    g["experts bf16 (4 runs)"]   = 4 * (me("bf16", 2304) + me("bf16", 1152))
    fa = lambda kv: get("FLASH_ATTN_EXT", kv=kv, nb=n)
    g["FA swa (40 runs, kv 1024)"] = 40 * fa(1024)
    for d in (1024, 32768, 131072):
        g["FA full (8 runs, kv %d)" % d] = 8 * fa(d)
    base = sum(v for k, v in g.items() if "FA full" not in k or "1024)" in k)
    print("== n=%d tokens per step; sum of listed ops (full attention at kv 1024): %.1f ms" % (n, base / 1e3))
    for k, v in g.items():
        print("  %-30s %7.2f ms  %4.1f%%" % (k, v / 1e3, 100 * v / base))
