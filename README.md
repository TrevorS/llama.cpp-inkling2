# llama.cpp, Inkling 2 fork

This branch is upstream llama.cpp ([`9c003a97a`](https://github.com/ggml-org/llama.cpp/commit/9c003a97a)) plus 12 commits that make Inkling 2 Mini fast on Apple Metal. Speculative output stays exact: with speculation on, the text is the same as plain decode, and a batch of 1-8 rows gives the same logits as a batch of one. The tools, the checks and the review notes are in [`scripts/inkling2`](scripts/inkling2/README.md).

## Changes from upstream

- `kv-cache`: when a sliding-window (SWA) tail is removed, continue at the first removed cell. Upstream reuses wrong cells after rejected drafts at the end of the ring.
- `graph`: layers on one device share the slot order of the conv state.
- `common`: size the rollback slots for n-gram drafts.
- `server`: do not start new drafts while more than one slot generates.
- `metal`: Inkling 2 kernels and fusions. Depth mix, short conv, loop controllers, concat + norm and other chains run as fused kernels: 2105 dispatches per decode step instead of 3541. NVFP4 mat-vec reads whole words. The NVFP4 MoE mat-mul keeps `half` operands (upstream uses `bfloat`, mean KLD 0.0016). Up to 8 queries, flash attention uses the dense mask, so each row is equal to a batch of one. `GGML_METAL_N_CB` overrides the number of command buffers.
- `metal`: binary ops split a long row over threadgroups. A 200k-wide add of the draft ran on one threadgroup (0.3 ms each).
- `inkling2`: keep the depth mix layout and the node order that the Metal fusions need.
- `dspark`: greedy drafts take only the argmax chain (`llama_set_draft_draws`). Before, each draft step built the sampled draws too: three top-k over the whole vocabulary that greedy drafts never read.
- `scripts/inkling2`: serving (`serve.sh`: DSpark draft k=3 plus n-gram drafts by default), exactness checks, KLD, benches, sweeps and profilers. The draft runs with one CPU thread (12 threads cost 1.7 ms per draft to start) and its own Q8_0 lm head (`draft-head.py`; the shared BF16 head of the target is twice the bytes).

Taken from upstream instead of the old fork: the DSpark/DFlash draft code, the chunked command buffers (2.7% faster decode than our 8 buffers), the NVFP4 type and the indexed bias in flash attention.

## Results (M4 Max, Inkling 2 Mini UD-Q8_K_XL)

| | upstream | this branch |
|---|---|---|
| pp512, depth 0 / 8k / 32k (tok/s) | 593 / 360 / 195 | 900 / 742 / 581 |
| tg128, depth 0 / 8k / 32k (tok/s) | 37.0 / 34.8 / 31.9 | 47.6 / 45.3 / 42.2 |

Server decode with the default n-gram + k3 speculation (`spec-sweep.sh`, tok/s, the text is the same as with speculation off):

| | short prompts | edit prompts | replay, 3 turns |
|---|---|---|---|
| speculation off | 46.9 | 45.8 | 44.9 |
| n-gram + k3 | 64.3 | 111.9 | 67.9 |

Checks: decode logits bitwise equal to the old fork and to the previous rebase, prefill KLD 0, `exact-matrix.sh` 27/27, `test-backend-ops` 20423/20423, `identity.sh` equal to plain decode.

## Next

A speculative step (k3, short prompts) is now about 39 ms: about 33 ms to verify 4 tokens and about 6 ms to draft.

1. The 4-token verify step. With duplicate expert routes made free, pp4 goes up only 8% (pp8 20%), so a MoE kernel that decodes each expert weight once for all its tokens can get part of that.
2. Store the experts of layers 20-23 as MXFP8. The BF16 weights are exact in E4M3 x E8M0 (4096 of 4096 sampled blocks), so the change is lossless and halves their bytes.
3. Fewer barriers. A decode step has 1701 barriers in 2105 dispatches, about 2 us each (16% of decode). Each norm that moves into the next mat-vec removes one.
4. Tune the BF16 K/V and NVFP4 down projection mat-vecs (341 and 349 GB/s).
5. The draft: 3 layers, the Q8_0 head (about 1 ms) and the feature inject graph each step; the DFlash2 drafter (upstream #33049).

Measured and dropped: experts read once through the cache (slower: one threadgroup does the tokens one after another), more threadgroups for the depth mix kernel (its own work is 1% of decode, the rest is dispatch cost), a Q4_0 draft head and a Q8_0 `markov_w2` (no gain).

---

# llama.cpp

![llama.cpp](https://raw.githubusercontent.com/ggml-org/llama-brand/refs/heads/master/logo/svg/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v1)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[v1.0.0](https://huggingface.co/blog/ggml-org/llama-cpp-v1) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Description

The main goal of `llama.cpp` is to enable LLM (and MLLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud.

The `llama.cpp` project is built on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Llama Chat, the built-in chat app, against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Llama Chat, the built-in chat app, against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [Text-to-speech (TTS) models](docs/multimodal/tts.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
