#!/bin/zsh
# Inkling 2 Mini on llama.cpp, tuned for agent use; model files in INKLING_DIR (default ~/models/Inkling-2-Mini-GGUF)
# DSpark k=3 with n-gram drafts is on by default: same output as no speculation, faster on agent prompts. SPEC=0 turns it off.
# 4 slots share one 131k KV pool (NP=1 for one slot); drafts run only while one slot generates.
# DRAFT=<file> sets the draft model; the default is Unsloth's mainline-format DSpark file.
# n-gram drafts of up to 7 tokens cover text the reply repeats, as in file edits; batches stay within 8 tokens. NGRAM=0 turns them off.
M=${INKLING_DIR:-$HOME/models/Inkling-2-Mini-GGUF}
spec=()
if [[ ${SPEC:-1} == 1 ]]; then
  spec=(-md ${DRAFT:-$M/dspark-Inkling-2-Mini-Q8_0.mainline.gguf} --spec-type draft-dspark --spec-draft-n-max ${SPEC_N:-3})
  if [[ ${NGRAM:-1} == 1 ]]; then
    spec+=(--spec-type ngram-mod,draft-dspark --spec-ngram-mod-n-max 7 --spec-ngram-mod-n-min 2 --spec-ngram-mod-n-match 24)
  fi
fi
exec ${0:A:h:h:h}/build/bin/llama-server \
  -m $M/Inkling-2-Mini-UD-Q8_K_XL.upstream.gguf \
  --mmproj $M/mmproj-F16.gguf \
  $spec \
  --alias inkling-2-mini \
  -c ${CTX:-131072} -np ${NP:-4} -kvu -fa on --jinja \
  -ctxcp 32 -cms 1024 \
  -n 32768 \
  --metrics \
  --host ${LLAMA_HOST:-127.0.0.1} --port ${LLAMA_PORT:-8080} \
  "$@"
