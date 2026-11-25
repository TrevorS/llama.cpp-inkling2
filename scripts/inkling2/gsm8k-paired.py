# paired comparison of two gsm8k.py runs on the items both answered: accuracy, discordant pairs, exact McNemar p, 95% CI of B - A
# usage: python3 gsm8k-paired.py <a.jsonl> <b.jsonl>
import json, math, sys

def load(p):
    return {r["i"]: r for r in map(json.loads, open(p))}

a, b = load(sys.argv[1]), load(sys.argv[2])
ids = sorted(set(a) & set(b))
n = len(ids)
acc_a = sum(a[i]["ok"] for i in ids) / n
acc_b = sum(b[i]["ok"] for i in ids) / n
n01 = sum(1 for i in ids if not a[i]["ok"] and b[i]["ok"])  # only B right
n10 = sum(1 for i in ids if a[i]["ok"] and not b[i]["ok"])  # only A right
k, m = min(n01, n10), n01 + n10
p = min(1.0, 2*sum(math.comb(m, j) for j in range(k + 1))/2**m) if m else 1.0
d = (n01 - n10) / n
se = math.sqrt(max(n01 + n10 - (n01 - n10)**2/n, 0)) / n
print(f"items {n}  A {acc_a*100:.2f}%  B {acc_b*100:.2f}%  B-A {d*100:+.2f} pp  95% CI [{(d-1.96*se)*100:+.2f}, {(d+1.96*se)*100:+.2f}]")
print(f"discordant: only A right {n10}, only B right {n01}  exact McNemar p = {p:.3f}")
print(f"mean tokens A {sum(a[i]['n'] for i in ids)/n:.0f}  B {sum(b[i]['n'] for i in ids)/n:.0f}")
