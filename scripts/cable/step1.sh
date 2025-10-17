# (optional) quick peek
python tools/peek_meta.py data/MVTec-AD

# filter to the 'cable' class  → writes array-style JSONs
python tools/mvtec_filter_onecls.py \
  --src data/MVTec-AD \
  --out data/_onecls/mvtec_cable \
  --cls cable

# convert array JSON → JSONL (one JSON per line; what the loader expects)
python tools/json_array_to_jsonl.py data/_onecls/mvtec_cable/train.json
python tools/json_array_to_jsonl.py data/_onecls/mvtec_cable/test.json

# sanity check (should print a JSON object line, not “[”)
head -n 2 data/_onecls/mvtec_cable/train.json
head -n 2 data/_onecls/mvtec_cable/test.json
