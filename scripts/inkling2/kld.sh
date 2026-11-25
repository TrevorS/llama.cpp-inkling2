#!/bin/zsh
# KLD and perplexity of a build against a reference build on the docs text (the check for rounding-level changes)
# usage: kld.sh <reference build/bin> <test build/bin> [chunks, default 6] [ENV=val ... for the test run]
# the reference logits are cached in $INKLING_DIR/bench/kld-<name of the reference dir>-<chunks>.bin; delete the file after a reference rebuild
# noise floor on this model: two attention implementations differ by mean KLD ~0.0013, Metal vs the CPU backend ~0.002
# prompt processing on Metal is not bit-reproducible run to run (max KLD ~5e-5 between two runs of one build); for bitwise checks use decode-dump.cpp
S=${0:A:h}
M=${INKLING_DIR:-$HOME/models/Inkling-2-Mini-GGUF}
G=$M/Inkling-2-Mini-UD-Q8_K_XL.upstream.gguf
E=$M/bench/eval.txt
ref=$1; tst=$2; chunks=${3:-6}; shift; shift; [[ $# -gt 0 ]] && shift
base=$M/bench/kld-${${ref:h:h}:t}-$chunks.bin

if [[ ! -f $base ]]; then
  $ref/llama-perplexity -m $G -f $E -c 2048 -fa on --chunks $chunks --kl-divergence-base $base 2>&1 | grep "Final estimate"
fi
env "$@" $tst/llama-perplexity -m $G -f $E -c 2048 -fa on --chunks $chunks --kl-divergence-base $base --kl-divergence 2>&1 \
  | grep -E "Mean PPL\(Q\)|Mean PPL\(base\)|Mean    KLD|Maximum KLD|99.9%   KLD|Same top p"
