# llama.cpp, Inkling 2 fork

This branch is upstream llama.cpp ([`9c003a97a`](https://github.com/ggml-org/llama.cpp/commit/9c003a97a)) plus 17 commits that make Inkling 2 Mini fast on Apple Metal. Speculative output stays exact: with speculation on, the text is the same as plain decode, and a batch of 1-8 rows gives the same logits as a batch of one. The tools, the checks and the review notes are in [`scripts/inkling2`](scripts/inkling2/README.md).

## Changes from upstream

- `kv-cache`: when a sliding-window (SWA) tail is removed, continue at the first removed cell. Upstream reuses wrong cells after rejected drafts at the end of the ring.
- `graph`: layers on one device share the slot order of the conv state.
- `common`: size the rollback slots for n-gram drafts.
- `server`: do not start new drafts while more than one slot generates.
- `metal`: Inkling 2 kernels and fusions. Depth mix, short conv, loop controllers, concat + norm and other chains run as fused kernels: 2105 dispatches per decode step instead of 3541. NVFP4 mat-vec reads whole words. The NVFP4 MoE mat-mul keeps `half` operands (upstream uses `bfloat`, mean KLD 0.0016). Up to 8 queries, flash attention uses the dense mask, so each row is equal to a batch of one. `GGML_METAL_N_CB` overrides the number of command buffers.
- `metal`: binary ops split a long row over threadgroups. A 200k-wide add of the draft ran on one threadgroup (0.3 ms each).
- `inkling2`: keep the depth mix layout and the node order that the Metal fusions need.
- `ggml`: the MXFP8 type (E4M3 values, one E8M0 scale per 32), from the open upstream #26157 with 128-value blocks and type 44 (fork only, never written to a file).
- `metal`: MXFP8 mat-vec and mat-mul with the sums of the BF16 kernels, so MXFP8 weights that come from BF16 give the same results bitwise.
- `llama`: weights converted at load, the model files stay as published. BF16 expert weights that are exact in MXFP8 are held as MXFP8, outside the mapped file (Inkling 2 layers 20-23: 7.6 GiB -> 3.9 GiB; `LLAMA_MXFP8=0` turns it off). A draft that shares the target's lm head gets a Q8_0 copy of it (`llama_model_copy_lm_head`; `LLAMA_DRAFT_HEAD_Q8=0` turns it off).
- `dspark`: greedy drafts take only the argmax chain (`llama_set_draft_draws`). Before, each draft step built the sampled draws too: three top-k over the whole vocabulary that greedy drafts never read.
- `scripts/inkling2`: serving (`serve.sh`: DSpark draft k=3 plus n-gram drafts by default), exactness checks, KLD, benches, sweeps and profilers. The draft runs with one CPU thread (12 threads cost 1.7 ms per draft to start). Prompts are read in ubatches of 2048 tokens (12% faster than 512 on long prompts, mean KLD 0.0004).

Taken from upstream instead of the old fork: the DSpark/DFlash draft code, the chunked command buffers (2.7% faster decode than our 8 buffers), the NVFP4 type and the indexed bias in flash attention.

## Results (M4 Max, Inkling 2 Mini UD-Q8_K_XL)

| | upstream | this branch |
|---|---|---|
| pp512, depth 0 / 8k / 32k (tok/s) | 593 / 360 / 195 | 900 / 742 / 581 |
| tg128, depth 0 / 8k / 32k (tok/s) | 37.0 / 34.8 / 31.9 | 47.6 / 45.3 / 42.2 |
| tg128 / pp4 with the MXFP8 experts (tok/s) | | 48.0 / 124.6 (BF16: 47.3 / 121.9) |
| pp4096, ubatch 512 / 2048 (tok/s) | | 787 / 879 |

Server decode with the default n-gram + k3 speculation (`spec-sweep.sh`, tok/s, the text is the same as with speculation off):

| | short prompts | edit prompts | replay, 3 turns |
|---|---|---|---|
| speculation off | 47.7 | 46.5 | 45.6 |
| n-gram + k3 | 65.0 | 113.3 | 68.6 |

Checks: decode logits bitwise equal to the old fork, to the previous rebase and with or without MXFP8, `exact-matrix.sh` 27/27, `test-backend-ops` 20663/20663, `identity.sh` equal to plain decode. The model takes 3.7 GiB less memory with the MXFP8 experts (server RSS 22.75 GiB, was 26.4).

## Next

A speculative step (k3, short prompts) is about 39 ms: about 32 ms to verify 4 tokens and about 5 ms to draft.

1. Fewer barriers. A decode step has 1701 barriers in 2105 dispatches, about 2 us each. With every norm-to-mat-vec barrier dropped (a timing test with wrong output) decode is 4.2% faster and pp4 2%; a norm folded into the BF16 and F32 mat-vecs has to redo the norm sum in each threadgroup, so expect about half.
2. Tune the BF16 K/V and NVFP4 down projection mat-vecs (341 and 349 GB/s).
3. The draft: the DFlash2 drafter (upstream #33049).

Measured and dropped: experts read once through the cache (slower), a multi-token NVFP4 MoE mat-vec that decodes each weight once for all tokens of an expert (17-25% slower: register pressure), more threadgroups for the depth mix kernel (1% of decode), a Q4_0 draft head and a Q8_0 `markov_w2`, backend sampling for the target and 2 command buffers (no gain).

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
