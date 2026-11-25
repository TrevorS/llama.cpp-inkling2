# GSM8K against a running llama-server: greedy, reasoning effort low, 4 concurrent requests, one jsonl line per item
# a run resumes from the items already in <out.jsonl>; a request that fails is retried 3 times and then left out
# usage: python3 gsm8k.py <port> <gsm8k.jsonl> <out.jsonl> [n_items]; compare two runs with gsm8k-paired.py
# note: 4 concurrent requests share batches, so a run is not bit-reproducible; the paired test handles that noise
import json, re, sys, threading, time, urllib.request

port, data, out = sys.argv[1], sys.argv[2], sys.argv[3]
items = [json.loads(l) for l in open(data)]
if len(sys.argv) > 4:
    items = items[:int(sys.argv[4])]

done = {}
try:
    for l in open(out):
        r = json.loads(l)
        done[r["i"]] = r
except FileNotFoundError:
    pass

lock = threading.Lock()
f = open(out, "a")
PROMPT = "Solve this math problem. End your reply with a final line of the form 'Answer: <number>'.\n\n{q}"

def num(s):
    m = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", s) or re.findall(r"(-?[\d,]*\.?\d+)", s)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None

def run(idx):
    it = items[idx]
    body = {"messages": [{"role": "user", "content": PROMPT.format(q=it["q"])}], "max_tokens": 2048, "temperature": 0, "reasoning_effort": "low"}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    text = r["choices"][0]["message"].get("content") or ""
    pred = num(text)
    ok = pred is not None and abs(pred - float(it["gold"])) < 1e-6
    rec = {"i": idx, "ok": ok, "pred": pred, "gold": it["gold"], "n": r["timings"]["predicted_n"], "text": text[-400:]}
    with lock:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        done[idx] = rec

todo = [i for i in range(len(items)) if i not in done]
t0 = time.time()
nxt = [0]

def worker():
    while True:
        with lock:
            if nxt[0] >= len(todo):
                return
            i = todo[nxt[0]]
            nxt[0] += 1
        for _ in range(3):
            try:
                run(i)
                break
            except Exception:
                time.sleep(2)

ts = [threading.Thread(target=worker) for _ in range(4)]
[t.start() for t in ts]
[t.join() for t in ts]
acc = sum(r["ok"] for r in done.values()) / max(1, len(done))
toks = sum(r["n"] for r in done.values())
print(f"{len(done)} items  accuracy {acc*100:.2f}%  mean tokens {toks/max(1, len(done)):.0f}  wall {time.time()-t0:.0f}s")
