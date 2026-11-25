# compare two replay.py outputs turn by turn: the text, and with LOGPROBS=1 the tokens and the log-probabilities of the top 3
# equal log-probabilities mean equal logits at every step; a difference shows the first token where the two runs part
# usage: python3 identity.py <ref.json> <other.json>; exit code 1 if anything differs
import json, sys
a, b = (json.load(open(f)) for f in sys.argv[1:3])
bad = len(a) != len(b)
for t, (x, y) in enumerate(zip(a, b)):
    lx, ly = (x.get("lp"), y.get("lp")) if "msg" in x else (None, None)
    if not lx or not ly:
        same = x.get("msg", x) == y.get("msg", y)
        print("turn %d: text %s" % (t, "same" if same else "DIFFERS"))
        bad |= not same
        continue
    key = lambda u: (u["token"], u["logprob"], [v["logprob"] for v in u["top_logprobs"]])
    n = min(len(lx), len(ly))
    tok = next((i for i in range(n) if lx[i]["token"] != ly[i]["token"]), None)
    lp  = next((i for i in range(n) if key(lx[i]) != key(ly[i])), None)
    if lp is None and len(lx) == len(ly):
        print("turn %d: %d tokens, tokens and log-probabilities same" % (t, n))
        continue
    bad = True
    print("turn %d: %d/%d tokens, first log-probability difference at token %s, first token difference at %s" % (t, len(lx), len(ly), lp, tok))
    for i in sorted({v for v in (lp, tok) if v is not None}):
        for name, l in (("ref  ", lx), ("other", ly)):
            print("   %s %4d %-14r %.9g %s" % (name, i, l[i]["token"], l[i]["logprob"], [(u["token"], round(u["logprob"], 6)) for u in l[i]["top_logprobs"]]))
sys.exit(1 if bad else 0)
