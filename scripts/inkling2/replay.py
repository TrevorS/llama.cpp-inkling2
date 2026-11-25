# replay a long multi-turn agent session against a running llama-server and save every reply
# turn 0 sends about ctx_chars of repo source (a 150000-char blob is about 56k tokens), turns 1-3 are short follow-ups
# usage: python3 replay.py <port> <out.json> <ctx_chars> [temperature] [image_path]; TURNS=n limits the turns
# env: LOGPROBS=1 saves the log-probabilities of every token (compare two runs with identity.py), STRIP=1 does not send the reasoning back
# (most clients do this; the server then restores a checkpoint each turn), MAXTOK=n sets max_tokens (default 400)
# compare two runs with: python3 -c 'import json,sys; a,b=[[t["msg"] for t in json.load(open(f))] for f in sys.argv[1:]]; print(a==b)' a.json b.json
import base64, json, os, pathlib, sys, urllib.request

port, out, ctx_chars = sys.argv[1], sys.argv[2], int(sys.argv[3])
temp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
image = sys.argv[5] if len(sys.argv) > 5 else None

# the blob is read from the repo this script lives in, so it changes with the code: compare runs of one checkout only
root = pathlib.Path(__file__).resolve().parents[2]
files = ["src/models/dspark.cpp", "common/speculative.cpp", "ggml/src/ggml-metal/ggml-metal-ops.cpp", "tools/server/server-context.cpp"]
files += sorted(str(f.relative_to(root)) for f in (root / "src").glob("*.cpp"))
blob = ""
for f in files:
    blob += f"\n===== {f} =====\n" + (root / f).read_text(errors="replace")
blob = blob[:ctx_chars]

system = "You are a coding agent working in the llama.cpp repository. Answer precisely and cite functions by name."
turns = [
    "Here are some source files from the repo:\n" + blob + "\n\nSummarize what the DSpark draft path does, step by step.",
    "Which function decides how many draft tokens are verified per step, and what limits it?",
    "Write a short C++ unit test idea for the chain trim in graph_inkling2. Keep it under 30 lines.",
    "Now list three risks in common_speculative_impl_draft_dflash::process and how to test each.",
]
turns = turns[:int(os.environ.get("TURNS", len(turns)))]

msgs: list[dict] = [{"role": "system", "content": system}]
log = []
for i, t in enumerate(turns):
    if image and i == 1:
        b64 = base64.b64encode(open(image, "rb").read()).decode()
        msgs.append({"role": "user", "content": [{"type": "text", "text": "Describe this image briefly, then: " + t},
                                                  {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]})
    else:
        msgs.append({"role": "user", "content": t})
    body = {"messages": msgs, "max_tokens": int(os.environ.get("MAXTOK", 400)), "temperature": temp, "seed": 42}
    if os.environ.get("LOGPROBS"):
        body.update({"logprobs": True, "top_logprobs": 3})
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
    m = r["choices"][0]["message"]
    tm = r["timings"]
    log.append({"msg": m, "timings": tm, "lp": (r["choices"][0].get("logprobs") or {}).get("content")})
    keep = m.get("reasoning_content") and not os.environ.get("STRIP")
    msgs.append({"role": "assistant", "content": m.get("content") or "", **({"reasoning_content": m["reasoning_content"]} if keep else {})})
    print("turn %d: prompt %d (cached %d) %.0f t/s, gen %d %.1f t/s%s" % (i, tm["prompt_n"], tm.get("cache_n", 0), tm["prompt_per_second"],
          tm["predicted_n"], tm["predicted_per_second"],
          "  draft %d/%d" % (tm["draft_n_accepted"], tm["draft_n"]) if tm.get("draft_n") else ""), flush=True)
json.dump(log, open(out, "w"), indent=1)
