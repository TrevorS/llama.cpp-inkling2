# write the GSM8K test set (1319 items) as jsonl lines {"q": question, "gold": final number} for gsm8k.py
# usage: curl -sL -o test.parquet https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet
#        uvx --with pandas --with pyarrow python3 -I gsm8k-prep.py test.parquet gsm8k.jsonl
import json, sys
import pandas as pd

d = pd.read_parquet(sys.argv[1])
with open(sys.argv[2], "w") as f:
    for q, a in zip(d["question"], d["answer"]):
        f.write(json.dumps({"q": q, "gold": a.split("####")[-1].strip().replace(",", "")}) + "\n")
print(len(d), "items")
