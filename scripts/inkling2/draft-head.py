# give the DSpark draft file its own quantized lm head, cut to the vocabulary, instead of the target's BF16 output.weight
# the draft reads the head on every draft step: Q8_0 halves the bytes, and the head no longer needs the copy that drops the padded rows
# drafts can change (rounding of the draft logits), the output cannot: speculation verifies every token against the target
# usage: uv run --with numpy --with tqdm python3 scripts/inkling2/draft-head.py <draft.gguf> <target.gguf> <out.gguf> [q8_0|q4_0]
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0] + "/gguf-py")
import gguf  # noqa: E402
from gguf.quants import quantize  # noqa: E402

src, tgt, dst = sys.argv[1:4]
qtype = {"q8_0": gguf.GGMLQuantizationType.Q8_0, "q4_0": gguf.GGMLQuantizationType.Q4_0}[sys.argv[4] if len(sys.argv) > 4 else "q8_0"]

r = gguf.GGUFReader(src)
arch = r.fields["general.architecture"].contents()
assert not any(t.name == "output.weight" for t in r.tensors), "the draft already has an lm head"
n_vocab = len(r.fields["tokenizer.ggml.tokens"].contents())

head = next(t for t in gguf.GGUFReader(tgt).tensors if t.name == "output.weight")
assert head.tensor_type == gguf.GGMLQuantizationType.BF16, head.tensor_type
n_embd = int(head.shape[0])
bits = np.frombuffer(head.data, dtype=np.uint16).reshape(-1, n_embd)[:n_vocab]
w = (bits.astype(np.uint32) << 16).view(np.float32)
print(f"head {w.shape[1]} x {w.shape[0]} (target rows {int(head.shape[1])}) -> {qtype.name}")

out = gguf.GGUFWriter(dst, arch, endianess=r.endianess)
for field in r.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
        continue
    vtype = field.types[0]
    sub = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
    out.add_key_value(field.name, field.contents(), vtype, sub_type=sub)

for t in r.tensors:
    out.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
out.add_tensor("output.weight", quantize(w, qtype), raw_dtype=qtype)

out.write_header_to_file()
out.write_kv_data_to_file()
out.write_tensors_to_file(progress=True)
out.close()
