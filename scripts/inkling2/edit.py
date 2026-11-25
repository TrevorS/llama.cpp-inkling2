# send 4 edit prompts (rewrite a given file with a small change) one by one to a running llama-server
# the replies repeat most of the input, which is the case n-gram drafts are made for
# prints decode tok/s and draft acceptance per prompt, saves the replies; equal reply files mean equal text
# usage: python3 edit.py <port> <out.json> [temperature]; the files are read from this checkout, so compare runs of one checkout only
# with a temperature above 0 the seed is fixed, so two settings that give the same logits give the same text
import json, pathlib, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
temp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
root = pathlib.Path(__file__).resolve().parents[2]
def src(path, first=None):
    lines = (root / path).read_text().splitlines()
    return "\n".join(lines[:first] if first else lines)
tasks = [
    ("scripts/inkling2/par.py", None, "Rename the variable `res` to `results` everywhere."),
    ("scripts/inkling2/gsm8k.py", None, "Add a constant TIMEOUT = 600 near the top and pass it as the timeout of every urlopen call."),
    ("src/models/inkling2-loop.cpp", None, "Rename the local variable `prelude` to `prelude_out` in the graph constructor."),
    ("scripts/inkling2/gpu-clocks.md", 30, "Change every 'tok/s' to 'tokens/s'."),
]
tok = secs = acc = dr = 0
replies = []
for path, first, change in tasks:
    prompt = f"Here is the file {path}:\n\n```\n{src(path, first)}\n```\n\n{change} Print the full updated file in one code block and nothing else."
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 2500, "temperature": temp, "seed": 42, "reasoning_effort": "low"}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
    t = r["timings"]
    tok += t["predicted_n"]; secs += t["predicted_ms"] / 1000
    acc += t.get("draft_n_accepted", 0); dr += t.get("draft_n", 0)
    replies.append(r["choices"][0]["message"])
    print("  %-32s gen %4d tokens %.1f tok/s, draft %d/%d" % (path, t["predicted_n"], t["predicted_per_second"], t.get("draft_n_accepted", 0), t.get("draft_n", 0)), flush=True)
print("edit: gen %.1f tok/s over %d tokens, draft %d/%d" % (tok / secs, tok, acc, dr))
json.dump(replies, open(out, "w"))
