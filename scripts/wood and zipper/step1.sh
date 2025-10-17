# (optional) peek structure
python tools/peek_meta.py data/MVTec-AD

# --- filter per class (creates array-style JSONs) ---
python tools/mvtec_filter_onecls.py --src data/MVTec-AD --out data/_onecls/mvtec_wood   --cls wood
python tools/mvtec_filter_onecls.py --src data/MVTec-AD --out data/_onecls/mvtec_zipper --cls zipper

# --- convert array JSON -> JSONL (one object per line; what loader expects) ---
python tools/json_array_to_jsonl.py data/_onecls/mvtec_wood/train.json
python tools/json_array_to_jsonl.py data/_onecls/mvtec_wood/test.json
python tools/json_array_to_jsonl.py data/_onecls/mvtec_zipper/train.json
python tools/json_array_to_jsonl.py data/_onecls/mvtec_zipper/test.json

# --- merge the two classes into a single meta (still JSONL) ---
mkdir -p data/_onecls/mvtec_wood_zipper
cat data/_onecls/mvtec_wood/train.json   data/_onecls/mvtec_zipper/train.json > data/_onecls/mvtec_wood_zipper/train.json
cat data/_onecls/mvtec_wood/test.json    data/_onecls/mvtec_zipper/test.json  > data/_onecls/mvtec_wood_zipper/test.json

# sanity check: should print JSON objects (lines starting with '{')
head -n 2 data/_onecls/mvtec_wood_zipper/train.json
head -n 2 data/_onecls/mvtec_wood_zipper/test.json

# (optional) line counts
wc -l data/_onecls/mvtec_wood_zipper/train.json
wc -l data/_onecls/mvtec_wood_zipper/test.json
