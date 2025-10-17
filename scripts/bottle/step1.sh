# optional peek
python tools/peek_meta.py data/MVTec-AD

# filter to the 'bottle' class  -> writes array-style JSONs
python tools/mvtec_filter_onecls.py \
  --src data/MVTec-AD \
  --out data/_onecls/mvtec_bottle \
  --cls bottle

# convert array JSON -> JSONL (one JSON per line, what the loader expects)
python tools/json_array_to_jsonl.py data/_onecls/mvtec_bottle/train.json
python tools/json_array_to_jsonl.py data/_onecls/mvtec_bottle/test.json

# quick sanity check (should print { ... }, not “[” )
head -n 2 data/_onecls/mvtec_bottle/train.json
head -n 2 data/_onecls/mvtec_bottle/test.json
