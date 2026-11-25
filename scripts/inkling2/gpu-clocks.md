# GPU clocks, idle time and heat on the M4 Max (Inkling 2)

Measured on 2026-11-23: MacBook Pro (Mac16,6), M4 Max with a 40-core GPU, 64 GB, macOS 27.2, on AC (94 W adapter), Energy Mode automatic (`pmset powermode 0`). The model is the Q8 GGUF at fork commit `29959ab72`. All counters come from `soc-sample.py` (IOReport, no sudo).

## Summary

1. The GPU and the memory fabric lower their clocks when the GPU idles in many short gaps. Normal decode does not have such gaps and runs at the top clocks.
2. Heat is the real limit. Long prefills, and decode right after them, hit the thermal controller (CLTM), which caps the GPU clock. The fans stay near their minimum until the GPU is at about 90-97 C.
3. This explains the 10-25% swings between runs of the same benchmark. Only interleaved A/B runs, or runs from the same thermal state, are comparable.

## 1. Idle gaps lower the clocks

The op timeline probe (`op-timeline.patch`) puts every op in its own encoder, which leaves a gap of a few us after each of the 3,500 ops of a token. Normal builds have no such gaps.

| Decode, depth 0 | tok/s | GPU active | GPU P-state (of active time) | Fabric (AFR) state | DRAM mean |
|---|---|---|---|---|---|
| normal build | 43.3 | 97% (81% with model load) | P15 (top) 100% | P7 (top) 100% | 233 GB/s |
| probe build | 12.2 | 50% | P1-P4 mostly, P15 0% | P2 62% | 46 GB/s |
| 1 command buffer (`GGML_METAL_N_CB=1`) | 35.4 | 77% with model load | P15 98% | P7 98% | 166 GB/s |

- With the probe, the governor drops both the GPU and the fabric to their lowest states. The same mat-vec kernel ran at 348 GB/s at depth 0, where the gaps dominate, and at 436 GB/s at 32k depth, where attention keeps the GPU busier. So probe timings are only good as shares within one run, never as absolute cost.
- One command buffer makes the GPU wait for the CPU encoder, which costs 18% of decode speed, but the clocks stay at the top. These waits are long and few, so the governor does not react to them.
- Normal decode spends 3% of its time idle (between tokens) and is never clock limited. The remaining distance to the bandwidth limit comes from the kernels and the number of dispatches, not from clocks.

## 2. Heat caps the GPU clock

Decode at depth 0, 512 tokens, from cool and right after 45 s of prefill:

| State | GPU temp before | tok/s | Thermal cap (CLTM) |
|---|---|---|---|
| cool (idle 90 s) | 43-48 C | 43.5, 43.3 | none |
| right after 45 s of pp2048 | 97 C | 43.4, 43.3 | none |

Decode at depth 0 does not heat the chip enough to be capped, even when it starts hot.

Decode at 32k depth (each run is a 32k prefill and then 128 tokens; the counters cover the decode only), three runs in a row from cool:

| Run | tok/s | GPU P-state (of active time) | Time under a thermal cap | GPU temp after | Fans (rpm) |
|---|---|---|---|---|---|
| 1 (from 40 C) | 39.7 | P15 76%, P14 24% | 47% (cap P9-P10) | 91 C | 3,100-3,400 |
| 2 | 37.5 | P10 42%, P12 37%, P8 17% | 96% (cap P6-P8) | 86 C | 4,900-5,300 |
| 3 | 38.5 | P14 47%, P12 25%, P10 14% | 100% (cap P7-P10) | 92 C | 4,900-5,300 |

Long prefill, pp2048 12 times in a row: 671 +/- 39 tok/s against 771 +/- 11 from cool (-13%). The GPU spends 57% of its active time in P8 and 13% in P6, and the thermal cap is on for 90% of the run. The fabric also drops from P7 to P6 for half the time.

- The fans run at their minimum (2,300-2,500 rpm, max 7,826) until the GPU reaches about 90-97 C, and on automatic they stay near 5,000 rpm even when capped.
- The 94 W adapter is not the limit: the system drew 67-77 W in these runs.
- The server replay (56k-token prompt, then 4 turns of 400 tokens) ranged from 23.5 to 32 tok/s for one configuration. Each replay starts with a 56k prefill, so its decode starts hot and capped, by how much depends on the state the run starts in. This matches the table above, but I did not log the counters during those replays.

## What to do

- Benchmarks: interleave A/B runs, or let the machine cool for 2 minutes between runs, and log `soc-sample.py` next to the result. A run with a thermal cap is not comparable to one without.
- Worth testing, not yet measured: Energy Mode "High Power" (System Settings > Battery, or `sudo pmset -a powermode 2`), which runs the fans harder, and a fixed fan speed (`sudo mactop --fan-set max`, back with `sudo mactop --fan-auto`). Both should cut the caps on long prefills and on decode at depth.
- Code: fewer, longer kernels keep the GPU busy without gaps, so dispatch fusion helps twice. Profiling tools that add gaps change the clocks they measure.

## Tools

- `soc-sample.py -- <command>`: GPU and fabric P-state residency, DRAM traffic, thermal caps (CLTM) and GPU loss reasons while the command runs; `--tail <s>` keeps the last seconds (the decode after a prefill), `--states` prints every state.
- `gpu-timeline.patch` + `gpu-timeline.py`: GPU time against wall time per token (command buffer level).
- `op-timeline.patch` + `op-timeline.py`: GPU time and bytes per op (adds gaps, see section 1).
