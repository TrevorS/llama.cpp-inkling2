# Rewrite an Unsloth Inkling 2 Mini GGUF (arch inkling2 + loop keys) into upstream's inkling2_loop layout.
# usage: python3 scripts/inkling2-unsloth-to-upstream.py <in.gguf> <out.gguf>
import re
import sys

import gguf
import numpy as np
from tqdm import tqdm

src, dst = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(src)
f = {k: v.contents() for k, v in r.fields.items()}
assert f["general.architecture"] == "inkling2", f["general.architecture"]

n_slot = f["inkling2.block_count"]
start, end = f["inkling2.loop.core_start"], f["inkling2.loop.core_end"]
passes, rank = f["inkling2.num_loops"], f["inkling2.loop.gate_rank"]
n_phys = n_slot - (passes - 1) * (end - start)
assert n_phys == 1 + max(int(t.name.split(".")[1]) for t in r.tensors if t.name.startswith("blk.")), n_phys

ARCH = "inkling2_loop"
w = gguf.GGUFWriter(dst, ARCH, endianess=r.endianess)

loop_keys = {
    "inkling2.num_loops":       f"{ARCH}.loop.count",
    "inkling2.loop.core_start": f"{ARCH}.loop.start",
    "inkling2.loop.core_end":   f"{ARCH}.loop.end",
    "inkling2.loop.gate_rank":  f"{ARCH}.loop.gate_rank",
}
for field in r.fields.values():
    name = field.name
    if name == "general.architecture" or name.startswith("GGUF."):
        continue
    vtype = field.types[0]
    sub = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
    val = field.contents()
    if name in loop_keys:
        w.add_key_value(loop_keys[name], val, vtype)
        continue
    if name.startswith("inkling2."):
        name = ARCH + name[len("inkling2"):]
        if name.endswith(".block_count"):
            val = n_phys
        elif sub is not None and len(val) == n_slot:
            # Unsloth stores one entry per layer run, upstream one per physical layer
            runs = list(range(start)) + list(range(start, end)) * passes + list(range(end, n_phys))
            phys = [val[runs.index(p)] for p in range(n_phys)]
            assert all(val[i] == phys[p] for i, p in enumerate(runs)), name
            val = phys
    w.add_key_value(name, val, vtype, sub_type=sub)
w.add_loop_varying(False)

ctrl = {"attn": "q", "k": "k", "v": "v", "mlp": "ffn"}
tensors = []  # (name, data, tensor_type)
queries = {}
for t in r.tensors:
    if t.name in ("loop_res_query.weight", "loop_kv_res_query.weight"):
        assert t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16), t.tensor_type
        queries[t.name.split(".")[0]] = (np.asarray(t.data), t.tensor_type)
        continue
    name = re.sub(r"\.loop_(attn|k|v|mlp)_(down|gain|forget)\.", lambda m: f".loop_{ctrl[m[1]]}_ctrl_{m[2]}.", t.name)
    tensors.append((name, t.data, t.tensor_type))

# rows 2i, 2i+1 of each query bank belong to core layer start+i: (attn, mlp) and (k, v), upstream wants (q, k, v, ffn)
q, qtype = queries["loop_res_query"]
kv, _ = queries["loop_kv_res_query"]
assert q.shape == kv.shape == (2 * (end - start), f["inkling2.embedding_length"]), (q.shape, kv.shape)
for i in range(end - start):
    msg = np.ascontiguousarray(np.stack([q[2 * i], kv[2 * i], kv[2 * i + 1], q[2 * i + 1]]))
    tensors.append((f"blk.{start + i}.loop_msg_query.weight", msg, qtype))

for name, data, ttype in tensors:
    w.add_tensor_info(name, data.shape, data.dtype, data.nbytes, ttype)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for name, data, ttype in tqdm(tensors, desc="tensors"):
    w.write_tensor_data(data, tensor_endianess=r.endianess)
w.close()
print(f"wrote {dst}: {len(tensors)} tensors, {n_phys} layers, loop [{start}, {end}) x{passes}, rank {rank}")
