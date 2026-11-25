# Inkling 2 Mini on Metal: state of the work and what to check

Branch `inkling2-rebase` (private fork; `inkling2-dspark` is the state before the rebase), M4 Max 64 GB, model `Inkling-2-Mini-UD-Q8_K_XL.upstream.gguf` (dense and lm head BF16, experts NVFP4 in layers 0-19 and BF16 in 20-23, layers 6-17 run 3 times, 48 layer passes per token). This note is for a reviewer: what we changed, how we checked it, and where we are least sure.

## Rules the work follows

1. Speculative output must be byte-identical to plain decode for one slot. Batches of 1-8 tokens must give each row the same logits as a batch of one.
2. No silent accuracy trades. A change that alters rounding is called rounding-level and measured with KLD; everything from 2026-11-23 claims to be bitwise equal to the unfused graph.

## Rebase onto upstream `5a09582d8` (2026-11-25)

Upstream merged Inkling 2 DSpark (#32218) in a reworked form, Metal NVFP4 (#32825), chunked command buffers (#32947) and its own Inkling 2 graph changes (#32855, #32940, #32948, #33053). The branch is now upstream plus 7 commits; the old history is on `inkling2-dspark`.

- Dropped for upstream's version: the DSpark PR snapshot and our DSpark changes (the output * hidden_out product is now `markov_w2`, folded at conversion; the draft is skipped without a context state; the chain length follows the draft batch), `dspark-vocab.py`, `--spec-draft-p-min` without a confidence head (it cost more than it saved), the NVFP4 kernels from Lyxot's PR, the `inkling2_loop` expert scale load (#32704), 8 command buffers by default with the geometric split and the pipeline rwlock (upstream's chunked single-thread commit decodes 47.5 against 46.25 tok/s, prefill level).
- Kept: the SWA ring fix (upstream still has the bug), the shared conv slot order, ngram-mod rollback slots, the server draft policy, all Metal fusions and kernels, the depth mix layout and node order in `inkling2.cpp` (upstream's layout needs a CONT per mix input: 3541 dispatches per decode step against 2105).
- Changed on upstream's code: the controller-norm fusion also matches upstream's `control()` (CONCAT in dim 1, same memory layout); NVFP4 MoE mat-mat with half operands, not bfloat (bfloat: mean KLD 0.0016 against the fork); NVFP4 mat-vec with 4 rows per simdgroup and the UE4M3 decode through fp16 (upstream's kernel: same values, other rounding, 2% slower decode); flash attention takes the indexed relative bias for more than 8 queries only (the dense mask per head for up to 8 queries is faster and matches a batch of one).
- Draft file: Unsloth's mainline-format `dspark-Inkling-2-Mini-Q8_0.gguf` (504 MB, sha256 `3da00cf7...`), stored as `dspark-Inkling-2-Mini-Q8_0.mainline.gguf`.

Checks of the rebased build:

- Against the fork: decode logits bitwise equal on 9 runs (3 prompts; batch 1 with 2 rollback slots, batch 4 with 7, batch 8 with 7 in the server layout); prefill mean KLD 0, max 5.6e-5 (run-to-run prefill noise), PPL 2.9151 against 2.9142.
- `exact-matrix.sh`: 27/27 bitwise against batch 1, 8/8 rejected drafts across the ring end, 2105 dispatches per decode step (the census now runs with `GGML_METAL_N_CB=0`: one encoder for the whole graph; the old count of 2044 missed the first 256 nodes).
- `test-backend-ops -b MTL0`: 20359/20359.
- `identity.sh k2 k7 ng` with upstream's DSpark and the mainline draft: all the same as spec off.
- llama-bench, alternated with the fork from 50 C, two runs each: pp512 894/889 against 897/895 (depth 0), 737/737 against 743/751 (8k), 581/587 against 577/581 (32k); tg128 47.2/47.0 against 46.9/47.6, 45.4/45.9 against 45.7/45.7, 41.7/41.5 against 41.9/41.6. Upstream alone: pp512 471/314/177, tg128 35.9/34.2/30.9 at the same depths.

Speculation with the mainline draft (`spec-sweep.sh`, GPU at 50 C or below before each set, decode tok/s; text equal to spec off in every cell):

| Setting | `short.py` | `edit.py` | replay (3 turns, 8.4k prompt) |
|---|---|---|---|
| spec off | 47.0 | 45.8 | 44.8 |
| k=2 | 56.8 | 70.5 | 61.6 |
| k=3 | 56.7 | 78.9 | 65.2 |
| k=5 | 51.2 | 86.1 | 63.0 |
| k=7 | 44.3 | 87.4 | 56.4 |
| n-gram + k=2 | 56.7 | 102.8 | 61.7 |
| n-gram + k=3 (the default) | 56.4 | 106.2 | 65.5 |
| n-gram + k=5 | 51.2 | 105.0 | 62.3 |
| n-gram + k=7 | 44.2 | 100.7 | 56.7 |

n-gram + k=3 stays the default: best or level on all three sets. Longer drafts only pay on the edit prompts without n-gram drafts. The mainline draft accepts more than the old one at k=3 (`edit.py` 78.9 against 67.3 tok/s with the old draft file before the rebase).

The sections below describe the fork before the rebase.

## Results

Each milestone build benchmarked on 2026-11-23 under the same conditions (llama-bench, 1 run each, 60 s cool-down; data `perf-history.csv`, plot `perf-history.png`):

| | Nov 20 `2e1af98a1` | now `e9c93a2a9` |
|---|---|---|
| pp512 depth 0 / 8k / 32k | 574 / 351 / 183 | 898 / 734 / 592 |
| tg128 depth 0 / 8k / 32k | 30.4 / 28.8 / 26.4 | 47.1 / 44.6 / 39.7 |

Server (DSpark k=2, 4 slots): short agent prompts 45.7 tok/s spec off, 52.0 spec on; output text identical on/off and identical to the build before the 2026-11-23 fusions.

Accuracy: PPL on 4 chunks of the docs text 3.0934 (unchanged since the depth-mix fusion, which moved it from 3.0907, KLD 0.00142, the only rounding-level change kept). Metal vs CPU KLD 0.00197 (noise floor between two attention implementations is about 0.0013).

## Commits, newest first (only fork commits)

2026-11-23, all claimed bitwise:
- `e9c93a2a9` fuse the short conv with input-dependent taps (17 nodes -> 1). Also `GGML_METAL_FUSION_MAX` 16 -> 48 (one pattern) with a new `GGML_METAL_FUSION_CHAIN` 16 (back-to-back packing), and `ggml_metal_fusion_max` no longer caps the span of the first pattern.
- `613124783` SIGMOID + MUL (attention output gate), CONCAT + RMS_NORM (controller norm), one zero-sink FILL per graph instead of per layer.
- `00d7780c0` REPEAT + GET_ROWS + MUL (NVFP4 per-expert scale). Also fixes chaining in `ggml_metal_fusion_max`: chained patterns were cut at 16 raw nodes, which could put a command buffer boundary inside the MoE reduce group so it ran unfused with a different add order.
- `29959ab72` CONCAT + CPY/SET_ROWS + SSM_CONV + ADD + SCALE (short conv with its state store, both the plain and the rollback-slot store).
- `181abdc0e` SITU activation chain (9 nodes -> 1).
- `c9f51e5e3`, `e21b6a4c0` tools: `soc-sample.py` (IOReport counters, no sudo), `op-timeline.patch`/`.py` (per-op GPU time and bytes), `gpu-clocks.md` (clock, idle and heat findings). All tools are listed in `README.md`.

2026-11-20 to 11-22 (summarized; details in commit messages): NVFP4 mat-vec without a lookup table, small-batch float mat-vec work and decode kernels for 2-8 rows (rule 1), FA with in-kernel relative-position bias, tanh gate / relative-bias mask / L2-norm / conv-residual / depth-mix fusions, 8 command buffers by default, pipeline lookup under a rwlock, cached command buffer split, shared conv slot order, DSpark draft fixes, server draft policy with several slots, upstream merge `779609469`.

Parked on `inkling2-act-quant` (`f1bf7cbf1`): NVFP4/MXFP8 rounding of expert inputs as trained (bitwise equal to the vLLM reference). Result: PPL +0.17-0.19% worse, GSM8K tied on 820 paired items (96.83 vs 96.95%, McNemar p = 1.0), decode -4%. Not merged.

## Upstream merge `83caf6753` and what followed (2026-11-23, later)

- The relative-position bias in flash attention now uses upstream's `ggml_flash_attn_ext_add_indexed_bias` (#32740): a table plus host-computed indices `[n_kv, n_tokens]`, where the fork passed cell and query positions. ggml.h, ggml.c, the CPU kernels and the other backends are upstream's; the Metal kernels and the Inkling 2 graph read the indices. `LLAMA_FA_REL_DISABLE=1` still forces the per-head mask.
- Cost of the indices: compute buffers at 131k context grow by 0.53 GB (Metal 844 -> 1113 MiB, CPU 142 -> 406 MiB); the host fill takes about 20 ms per 512-token chunk at 64k depth (about 2% of the chunk, linear in depth). Speed at depth 0 and 32k is the same as before the merge; 64k and 131k end to end are not timed yet.
- The CPU backend runs the indexed bias (upstream reports it as not supported), so test-backend-ops compares Metal with it.
- Mat-mul templates follow upstream #32677; the NVFP4 mat-mat kernels keep half operands as before. Open upstream PR #32825 (Metal NVFP4) uses the same symbol names and will conflict.
- After the merge, not committed yet: the residual ADD after a plain short conv joins `kernel_conv_store` (87 fewer dispatches per token, 2131 -> 2044); DSpark skips the draft of a sequence without a context feature where it threw (the throw was not caught and would stop the server); rollback slots cover ngram-mod drafts (`need_n_rs_seq`), and `serve.sh NGRAM=1` adds n-gram drafts of up to 7 tokens.
- Checks: decode logits bitwise equal to the build before the merge for (BATCH, N_RS) = (1,0), (1,2), (3,2), (8,2), (3,0), (8,7); `exact-matrix.sh` 27/27; prefill mean KLD 0, max 5e-5; test-backend-ops -b MTL0 20155/20155 at the merge and 20160/20160 with the residual ADD fusion.
- Speculation text: spec off, DSpark k=2 and n-gram + DSpark give the same text on `short.py`, `edit.py` and a seeded temperature 0.8 replay. On `edit.py` the target decodes 2320 steps for 6639 tokens with DSpark and 1374 with n-gram + DSpark. Speeds of that run were taken under load and are not recorded here.
- Draft length 3 (`SPEC_N=3`): same text on the same three sets. Target steps on `edit.py`: 1796 with DSpark (k=2: 2320), 1230 with n-gram + DSpark (k=2: 1374); on `short.py` 1132 for 2876 tokens (k=2: 1293), with 52% of drafted tokens accepted against 62%. Each step verifies 4 rows where k=2 verifies 3, so the speed needs a quiet machine to decide.
- Ring bug found with draft length 7 (2026-11-23): the server keeps the SWA layers in a ring of cells (4608 with 4 slots). A draft batch that crossed the end of the ring and was then rejected left the write head at cell 0, so the next tokens went to other cells than in plain decode. The attention reads the cells in cell order, so the sums ran in another order: a rounding-level difference at first (1e-4 in a log-probability), which grew to 0.2 within 15 tokens (expert routing is discrete). Any draft length could hit it, about 1 in 8 ring crossings at k=2 by estimate. `llama_kv_cache::seq_rm` now continues at the cell of the first removed position when the tail of a sequence is removed from a SWA cache; plain decode is bitwise unchanged. The bitwise tools missed it because the library default is a full-size SWA cache; they now use the ring, and `exact-matrix.sh` has 8 cases of rejected drafts across the ring end.
- After the fix: spec off, k=2, k=7 and n-gram + k=2 give the same tokens and log-probabilities on the seeded temperature 0.8 replay (3 turns, crosses the ring end) and the same text on `edit.py` at temperature 0.8. Sampling with a fixed seed is the sensitive check: greedy text hid the bug.
- `identity.sh` (added after the ring fix): spec on vs off through the server at temperature 0.8 with a fixed seed, tokens and log-probabilities of every reply. k=2, k=7 and n-gram + k=2 equal spec off on 4 cases: reasoning sent back, reasoning stripped (a checkpoint restore each turn), replies that end by themselves (depth to 14k, two ring crossings), edit prompts.
- Draft cost (2026-11-23): a DSpark draft took 9.2 ms at k=2 and 25 ms at k=7, about 3 ms per draft token, because the head ran the 926 MB output matrix once per chain step (`bias = output * (hidden_out * latent + b)`). `dspark-vocab.py` writes `output * hidden_out` ([256, n_vocab], F32, 206 MB) into a copy of the draft file and `build_inkling2_head` uses it when present: 6.4 ms at k=2, 13.5 ms at k=7, same acceptance counts and same text. What is left per draft token: the 206 MB read and the first-maximum selection (about 0.4 ms, measured by replacing it with a plain argmax).
- `--spec-draft-p-min` now works for a draft without a confidence head: the draft ends at the first token whose probability is below the value (`SPEC_P`). The probability predicts acceptance (0.99 and above: 98% accepted, 65% of the verified draft tokens; below 0.5: under 30%), but the setting needs all 7 draft tokens scored at every step, and that costs more than it saves, see the table. Off by default.
- Timing with a cool-down (GPU at 60 C or below before every prompt set, one sample per cell, tok/s and the ratio to spec off):

  | Setting | `short.py` | `edit.py` | replay (3 turns, 8.4k prompt) |
  |---|---|---|---|
  | spec off | 45.6 | 40.3 | 34.9 |
  | k=2, old draft file (the default until now) | 49.2 (1.08) | 62.5 (1.55) | 51.5 (1.48) |
  | k=2, new draft file | 54.6 (1.20) | 67.3 (1.67) | 56.4 (1.62) |
  | k=7, cutoff 0.8, new draft file | 49.0 (1.08) | 83.2 (2.06) | 53.1 (1.52) |
  | n-gram + k=2, new draft file (the default now) | 55.3 (1.21) | 95.4 (2.37) | 57.4 (1.64) |
  | n-gram + k=7, cutoff 0.8, new draft file | 49.5 (1.09) | 95.3 (2.36) | 53.5 (1.53) |

- The same with a deeper cool-down (GPU at 50 C or below before every set) and two more settings:

  | Setting | `short.py` | `edit.py` | replay |
  |---|---|---|---|
  | spec off | 46.8 | 44.4 | 43.7 |
  | k=2, old draft file | 55.8 (1.19) | 67.8 (1.53) | 57.9 (1.32) |
  | n-gram + k=2, new draft file (the default) | 59.8 (1.28) | 102.5 (2.31) | 61.6 (1.41) |
  | n-gram + k=3, new draft file | 59.5 (1.27) | 105.5 (2.38) | 64.8 (1.48) |

  n-gram + k=3 was also ahead of n-gram + k=2 in the 60 C run (54.7 / 98.3 / 60.2 against 54.0 / 95.3 / 58.9): 3-5% on the two deep sets, level on `short.py`. serve.sh uses k=3 since then.
- More points for the default (60 C run, spec off against the default): temperature 1.0 with a fixed seed: `edit.py` 39.0 -> 82.0 tok/s, replay 35.4 -> 46.5; a 56k-token context (replay with 150000 chars): 29.5 -> 43.0. Text equal in all of them.
- Upstream merge at depth, builds alternated from 52 C: at 64k the current build does pp512 417 and 409 tok/s against 435 and 426 before the merge (about 4% less), tg64 is 35.8-35.9 for both. At 131k (one pair, both under a thermal cap): 249 against 258, tg64 23.8 against 22.2. A first pair at 64k showed 22% less; the current build ran second there and was capped.
- Heat decides run-to-run differences: without the cool-down the same setting moved by 15-25% with what ran before it (k=7 on `short.py`: 31 tok/s after other runs, 45.7 after one minute of rest; spec off on `short.py`: 45.8 first, 34.7 after 25 minutes of runs). Earlier ratios in this file that were taken back to back are not reliable for the GPU-heavy settings.
- Draft length of the drafter: trained with blocks of 8 (`mtp_config.block_size`), so 7 draft tokens at most. vLLM and SGLang recipes verify the full block (7 drafts) on data-center GPUs; Unsloth uses 2 for llama.cpp ("measured fastest"), the llama.cpp default is 3. Acceptance per draft position at k=7 on the replay: 0.79, 0.71, 0.58, 0.54, 0.46, 0.37, 0.29 (turn 1) down to 0.68, 0.49, 0.34, 0.22, 0.10, 0.07, 0.06 (turn 3). Target steps, k = 2 / 3 / 7: `short.py` 1293 / 1132 / 959, `edit.py` 2320 / 1796 / 1069, replay 500 / 426 / 335. A step at k verifies k + 1 rows, so the best k depends on the cost of one more row; not timed yet.
- Graph rebuilds: about 5 ms each (11,100 nodes), on about 1% of DSpark steps and 6% of n-gram steps, so a graph cache is not worth it.
- DSpark: the stall and the seed issue reported upstream are in `--spec-draft-sampling probabilistic` only, which is off by default and not set by `serve.sh`.
- Dispatch census of one decode step (`census.py`): 530 MUL_MAT, 288 controller dispatches (concat-norm, down projection and tanh gate for 96 controllers), 94 MUL_MAT_ID, 92 SET_ROWS (KV store), 53 RMS_NORM, 46 PAD of the constant bias table. Left as is: the PAD (46 dispatches, upstream pads per token too) and the controllers of Q, K and V in shared kernels (144 fewer). At about 2 us per dispatch these are 0.4% and 1.3-1.6% of a 21.5 ms token by estimate; the controller kernel would need the mat-vec arithmetic of three weight sets in one bitwise kernel.

## How "bitwise" was checked

- `scripts/inkling2/decode-dump.cpp`: decodes 48 prompt tokens in batches of `BATCH` with `N_RS` rollback slots and writes all logits; old and new builds compared with `cmp` for (BATCH, N_RS) = (1,0), (1,2), (3,2), (8,2), (3,0), and batch 3/8 against batch 1 with N_RS=2.
- `quick.sh`: batch-exact (7 tokens, logits and capture), greedy text vs reference, PPL.
- Server text with spec on vs off (8 prompts, 2876 tokens).
- `test-backend-ops -b MTL0`: 20042/20042, including 7 new fusion tests (`SITU_FUSE`, `CONV_STORE_FUSE`, `CONV_STORE_DYN_FUSE`, `EXPERT_SCALE_FUSE`, `SIGMOID_MUL_FUSE`, `CONCAT_NORM_FUSE`). These compare with the CPU within the usual NMSE, not bitwise with the unfused Metal path.
- Fused elementwise kernels keep each intermediate through `as_type<...>(as_type<uint...>(v) ^ args.zero)` (zero passed at run time) so fast-math cannot merge steps; without it the SITU kernel gave KLD 0.0012 against the unfused nodes.

## Where to look hardest

1. Fusion pattern checks in `ggml/src/ggml-metal/ggml-metal-fusion.cpp` (`check_situ`, `check_conv_store`, `check_conv_store_dyn`, `check_expert_scale`, `check_sigmoid_mul`, `check_concat_norm`): are the shape, stride, view and use-count conditions tight enough that no other model or batch shape can match and compute something else? The conv patterns depend on the raw node order that `src/models/inkling2.cpp` now forces with `ggml_build_forward_expand` (taps and conv expanded next to the state store).
2. `ggml_metal_fusion_max` and the new `GGML_METAL_FUSION_MAX` / `GGML_METAL_FUSION_CHAIN` split: packing, command buffer boundaries and `ggml_metal_fusion_add_alloc_deps` with 40-node patterns.
3. `kernel_conv_store` (misc.metal): reads the past state from the gathered history copy and writes the new state in the same kernel. This is safe only because `build_rs` gathers into a separate tensor; a custom `get_state_rows` would alias them. It also indexes a register array `xs[8]` dynamically.
4. Concurrency inside fused encoders: each one checks only its inner nodes against the open range set (`ggml_metal_op_concurrency_check`). `EXPERT_SCALE` is anchored at a REPEAT that depends only on a weight; the graph optimizer and the encoder handled it in all tested graphs.
5. Spec exactness was tested on one prompt and batch sizes 1, 3, 8. With `N_RS=0`, batch 3 and batch 1 differ, in the old build too; only the rollback path (what the server uses with speculation) is covered by rule 1.
6. Prefill paths (more than 8 tokens) were checked by PPL only, not bitwise.

## Known limits and open items

- Measurements: one run per point, 3-5% run-to-run noise. Long prefills and decode right after a 32k+ prefill are capped by the thermal controller (see `gpu-clocks.md`); the server replay swing (23.5-32 tok/s) is attributed to this by inference, not by counters during those runs.
- Per token we are at about 50% of the 546 GB/s bandwidth floor. About 2,200 dispatches remain; big mat-vecs run at about 74% of peak.
- Next candidates, all rounding-level: RMS norm folded into the following mat-vecs (about 150 per token), the controller norm plus its 64-row projection, a faster depth-mix kernel (one threadgroup per token, about 14 us per call, 120 calls). Kernel tuning of the 4.7 MB K/V projections and the NVFP4 down projection (about 340 GB/s) could give about 4%.
- Untested fixes for heat: High Power mode, fixed fan speed (both need sudo).
