# send N agent prompts at once to a llama-server and report aggregate and per-request speed
# usage: python3 par.py <port> <n_concurrent>
import json, sys, threading, time, urllib.request
port, n = sys.argv[1], int(sys.argv[2])
prompts = [
    "Write a Python function that parses an ISO 8601 duration string like P3DT4H5M into total seconds. Include type hints.",
    "Write a bash script that finds the 10 largest files under a directory, skipping .git.",
    "Explain how a B-tree insert splits nodes, step by step, in plain prose.",
    "Write a Rust function that merges two sorted Vec<i64> into one sorted Vec without extra crates.",
    "Write a SQL query that returns, per customer, the month with their highest total order amount.",
    "Explain what a write-ahead log is and why databases use one, in plain prose.",
    "Write a TypeScript debounce function with proper generic types.",
    "Write a Go HTTP handler that returns the server time as JSON.",
]
res = [None]*n
def run(i):
    body = {"messages": [{"role": "user", "content": prompts[i % len(prompts)]}], "max_tokens": 256, "temperature": 0, "reasoning_effort": "low"}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req))
    res[i] = (r["timings"], r["choices"][0]["message"]["content"])
t0 = time.time()
ts = [threading.Thread(target=run, args=(i,)) for i in range(n)]
[t.start() for t in ts]; [t.join() for t in ts]
wall = time.time() - t0
tok = sum(t["predicted_n"] for t, _ in res)
acc = sum(t.get("draft_n_accepted", 0) for t, _ in res); dr = sum(t.get("draft_n", 0) for t, _ in res)
per = [t["predicted_n"]/(t["predicted_ms"]/1000) for t, _ in res]
print("n=%d  total %.1f tok/s  per-request %.1f-%.1f tok/s  draft %d/%d  wall %.1fs" % (n, tok/wall, min(per), max(per), acc, dr, wall))
json.dump([c for _, c in res], open(sys.argv[3] if len(sys.argv) > 3 else "/dev/null", "w"))
