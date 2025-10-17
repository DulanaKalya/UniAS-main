# 1) filter only 'leather' samples → writes array-style JSONs
python tools/mvtec_filter_onecls.py \
  --src data/MVTec-AD \
  --out data/_onecls/mvtec_leather \
  --cls leather

# 2) convert array JSON → JSONL (one JSON object per line; what the loader expects)
python tools/json_array_to_jsonl.py data/_onecls/mvtec_leather/train.json
python tools/json_array_to_jsonl.py data/_onecls/mvtec_leather/test.json

# 3) quick sanity check (should print lines starting with “{”)
head -n 2 data/_onecls/mvtec_leather/train.json
head -n 2 data/_onecls/mvtec_leather/test.json
