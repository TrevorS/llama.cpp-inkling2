# send 8 short agent prompts one by one to a running llama-server; print decode tok/s and draft acceptance, save the replies
# usage: python3 short.py <port> <out.json>; two runs give the same text when their replies files are equal (cmp)
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
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
tok = secs = acc = dr = 0
replies = []
for p in prompts:
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": 400, "temperature": 0, "reasoning_effort": "low"}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})))
    t = r["timings"]
    tok += t["predicted_n"]; secs += t["predicted_ms"] / 1000
    acc += t.get("draft_n_accepted", 0); dr += t.get("draft_n", 0)
    replies.append(r["choices"][0]["message"])
print("short: gen %.1f tok/s over %d tokens, draft %d/%d" % (tok / secs, tok, acc, dr))
json.dump(replies, open(out, "w"))
