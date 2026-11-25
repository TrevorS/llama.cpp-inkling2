# Inkling 2 tools

Scripts for serving, checking and profiling Inkling 2 Mini on this fork (Metal, M4 Max). Model files live in `$INKLING_DIR` (default `~/models/Inkling-2-Mini-GGUF`); reference data and outputs in `$INKLING_DIR/bench`. Shell scripts are zsh. Scripts that start their own server stop the live one on port 8080 first, because two copies of the model do not fit in 64 GB, and start it again at the end.

Notes: `REVIEW.md` (state of the work, what to check), `gpu-clocks.md` (clocks, idle time, heat), `perf-history.csv` / `perf-history.png` (milestone benchmarks).

## Setup and serving

| Tool | What it does |
|---|---|
| `../inkling2-unsloth-to-upstream.py <in.gguf> <out.gguf>` | Rewrites an Unsloth Inkling 2 Mini GGUF into the `inkling2_loop` layout this fork loads (`*.upstream.gguf`). |
| `serve.sh` | The live server: 4 slots on one 131k KV pool, DSpark k=3 with n-gram drafts (`NGRAM=0` turns the n-gram drafts off), flash attention, context checkpoints. Env: `SPEC=0`, `SPEC_N`, `DRAFT` (draft file; the default is `dspark-Inkling-2-Mini-Q8_0.mainline.gguf`, Unsloth's mainline-format drafter), `NGRAM`, `NP`, `CTX`, `LLAMA_PORT`, `LLAMA_HOST`. |

## Correctness

| Tool | What it does |
|---|---|
| `quick.sh` | Everyday check: batch exactness (`batch-exact.cpp`), greedy text against a reference, short perplexity, pp512/tg128. `--base <bin dir>` interleaves the speed bench with another build, `--depth` adds 32k, `--agent` runs prompts through the server, `--ops OP` runs test-backend-ops. The first run writes the references. |
| `batch-exact.cpp` | K tokens decoded one by one vs as one batch; logits and named intermediate tensors must match bitwise (rule 1 of `REVIEW.md`). Built by `quick.sh`. |
| `decode-dump.cpp` | Writes every logit row of a decode to a file, `BATCH` tokens per step with `N_RS` rollback slots, so two builds or settings can be compared with `cmp`. Compare runs with the same token count. `DEPTH` decodes a prefix first, `KEEP`/`ROLL` reject part of every batch and remove it like a rejected draft, `SERVER=1` gives the 4 slot cache layout, `@file` reads the prompt from a file. The SWA cache is a ring as in the server (the library default is a full-size cache, which hides ring effects). Build: `c++ -std=c++17 -O2 -Wno-deprecated-declarations -Iinclude -Iggml/include scripts/inkling2/decode-dump.cpp -Lbuild/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/build/bin -o decode-dump`. |
| `kld.sh <ref bin> <test bin> [chunks] [ENV=val]` | Mean and max KLD, PPL and same-top-token rate of a build against a reference build on the docs text; for rounding-level changes. |
| `exact-matrix.sh [dispatches]` | Wide check of rule 1 after a kernel, graph or merge change: 3 prompts, batches of 1-8 with 7 rollback slots and of 1 and 3 with 2 slots, all bitwise against batch 1; then rejected drafts across the end of the SWA cache ring (one slot at depth 1480, server layout at depth 9150, 8 cases); then the dispatch census against the expected count (2105). |
| `census.py <log> [--check N]` | Metal dispatches of one decode step by op, from a `GGML_METAL_GRAPH_DEBUG=1` log of `decode-dump`. A fusion that stops matching shows up as more dispatches. |
| `short.py <port> <out.json>` | 8 short agent prompts, greedy; decode tok/s, draft acceptance and the replies. Equal reply files mean equal text, e.g. spec on vs off. |
| `edit.py <port> <out.json> [temperature]` | 4 edit prompts (print a given file with a small change), greedy by default; the replies repeat the input, the case for n-gram drafts. With a temperature the seed is fixed: a sensitive check that two settings give the same logits. |
| `replay.py <port> <out.json> <chars> [temperature]` | A 4-turn agent session with a long first prompt (150000 chars is about 56k tokens); prompt and decode speed per turn, and the replies. Env: `TURNS`, `MAXTOK`, `LOGPROBS=1` (save the log-probabilities of every token), `STRIP=1` (do not send the reasoning back, as most clients do). |
| `identity.sh [settings]` | Speculation on vs off through the server at temperature 0.8 with a fixed seed: tokens and log-probabilities of every reply must equal spec off. 4 cases (reasoning sent back, reasoning stripped, replies that end by themselves, edit prompts) for `k2 k7 ng` by default. The sensitive check of rule 1: sampling shows a logit difference within a few tokens, greedy text can hide it. `identity.py <ref.json> <other.json>` compares two replay outputs and prints the first token that differs. |
| `gsm8k-prep.py`, `gsm8k.py`, `gsm8k-serve.sh`, `gsm8k-paired.py` | Task accuracy A/B: prepare the GSM8K test set, run it against a server (greedy, 4 at a time, resumable), and compare two runs item by item (discordant pairs, exact McNemar p, 95% CI). |

## Speed

| Tool | What it does |
|---|---|
| `spec-sweep.sh [out dir]` | Spec off and DSpark K = 1..4 through the server, with `short.py` and `replay.py`, repeats K=2 and spec off to catch drift, and checks that every setting gives the spec-off text. |
| `par.sh <N> [ENV=val] -- [server args]`, `par.py` | N concurrent agent requests against a fresh server: total and per-request tok/s. |
| `history.sh <worktree> <out.csv> <commit...>`, `history-plot.py` | Builds each commit in a separate worktree and benchmarks pp512/tg128 at depth 0, 8k, 32k under the same conditions; plots the result (`uvx --with matplotlib python3 history-plot.py in.csv out.png`). |
| `mmperf.sh` | test-backend-ops perf of the BF16 mat-mul shapes of the model, best of 3 (needs `profile-cases.patch`). |
| `profile.py` | Estimated GPU time of one decode step per op group from test-backend-ops perf (needs `profile-cases.patch`). |

## Profiling probes

The patches add env-gated instrumentation; apply with `git apply`, build, and remove with `git apply -R` before committing.

| Tool | What it does |
|---|---|
| `gpu-timeline.patch` + `gpu-timeline.py` | `X_GPUT=1`: GPU time against wall time per token and the idle gaps between command buffers. |
| `op-timeline.patch` + `op-timeline.py` | `X_OPT=1`: GPU time and bytes moved per op (one encoder per op), grouped by op and shape, against the bandwidth floor. The probe adds gaps that lower the GPU clock, so read its times as shares within one run. |
| `soc-sample.py [--tail s] [--states] -- <command>` | GPU and fabric P-state residency, DRAM traffic, thermal caps (CLTM) and GPU loss reasons from IOReport while a command runs; no sudo. |
| `heat.sh [bin dir]` | The thermal runs of `gpu-clocks.md`: decode from cool and right after a long prefill, decode at 32k three times in a row, a long prefill alone (needs `mactop` for temperatures and fans). |

## Habits that matter on this machine

- Speed varies 3-5% from run to run, and 10-25% after long prefills (thermal caps): interleave A/B runs or let the machine cool between them.
- Prompt processing on Metal is not bit-reproducible between runs (max KLD about 5e-5). Bitwise checks use decode (`decode-dump.cpp`, `batch-exact.cpp`).
- `decode-dump` rows are `n_vocab` = 200058 floats; the output tensor is padded to 201024 rows.
