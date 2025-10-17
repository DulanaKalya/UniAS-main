# (optional) peek keys/paths
python tools/peek_meta.py data/MVTec-AD

# filter to the 'transistor' class  → writes array JSONs
python tools/mvtec_filter_onecls.py \
  --src data/MVTec-AD \
  --out data/_onecls/mvtec_transistor \
  --cls transistor

# convert array JSON → JSONL (one JSON per line; what the loader expects)
python tools/json_array_to_jsonl.py data/_onecls/mvtec_transistor/train.json
python tools/json_array_to_jsonl.py data/_onecls/mvtec_transistor/test.json

# quick sanity check (should start with "{", not "[")
head -n 2 data/_onecls/mvtec_transistor/train.json
head -n 2 data/_onecls/mvtec_transistor/test.json
