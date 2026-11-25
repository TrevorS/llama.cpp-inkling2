#!/bin/zsh
# time BF16 MUL_MAT at the Inkling shapes (k=2304/3072/4608), best of 3; apply profile-cases.patch and build test-backend-ops first
# usage: mmperf.sh [ENV=val ...]
cd ${0:A:h:h:h}
for r in 1 2 3; do
  env "$@" build/bin/test-backend-ops perf -o MUL_MAT -b MTL0 -p 'type_a=(bf16|f32),type_b=f32,m=[0-9]+,n=[148],k=(2304|3072|4608),bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,o=1,' 2>&1 | grep -E "us/run" | sed -E 's/.*type_a=([a-z0-9]+),.*m=([0-9]+),n=([0-9]+),.* ([0-9.]+) us\/run.*/\1.m\2.n\3 \4/'
done | awk '{ if (!($1 in m) || $2 < m[$1]) m[$1] = $2 } END { for (n in m) print n, m[n] }' | sort -t. -k1,1 -k2,2
