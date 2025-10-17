# peek original dataset
python tools/peek_meta.py data/MVTec-AD

# filter only toothbrush samples → writes train.json/test.json
python tools/mvtec_filter_onecls.py \
  --src data/MVTec-AD \
  --out data/_onecls/mvtec_toothbrush \
  --cls toothbrush

# convert array JSON → JSONL
python tools/json_array_to_jsonl.py data/_onecls/mvtec_toothbrush/train.json
python tools/json_array_to_jsonl.py data/_onecls/mvtec_toothbrush/test.json

# sanity check (should show JSON objects, not “[”)
head -n 2 data/_onecls/mvtec_toothbrush/train.json
head -n 2 data/_onecls/mvtec_toothbrush/test.json
