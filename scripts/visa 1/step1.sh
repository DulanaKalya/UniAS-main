# 1) Filter each class → array JSONs
for C in candle capsules cashew chewinggum fryum macaroni1 macaroni2; do
  python tools/mvtec_filter_onecls.py \
    --src data/VisA \
    --out data/_onecls/visa_${C} \
    --cls ${C}
done

# 2) Convert array JSON → JSONL (one object per line; what the loader expects)
for C in candle capsules cashew chewinggum fryum macaroni1 macaroni2; do
  python tools/json_array_to_jsonl.py data/_onecls/visa_${C}/train.json
  python tools/json_array_to_jsonl.py data/_onecls/visa_${C}/test.json
done

# 3) Merge selected classes into ONE meta (edit the list to use a subset)
mkdir -p data/_onecls/visa_multi
cat \
  data/_onecls/visa_candle/train.json \
  data/_onecls/visa_capsules/train.json \
  data/_onecls/visa_cashew/train.json \
  data/_onecls/visa_chewinggum/train.json \
  data/_onecls/visa_fryum/train.json \
  data/_onecls/visa_macaroni1/train.json \
  data/_onecls/visa_macaroni2/train.json \
  > data/_onecls/visa_multi/train.json

cat \
  data/_onecls/visa_candle/test.json \
  data/_onecls/visa_capsules/test.json \
  data/_onecls/visa_cashew/test.json \
  data/_onecls/visa_chewinggum/test.json \
  data/_onecls/visa_fryum/test.json \
  data/_onecls/visa_macaroni1/test.json \
  data/_onecls/visa_macaroni2/test.json \
  > data/_onecls/visa_multi/test.json

# 4) Quick sanity check (lines should start with "{", not "[")
head -n 2 data/_onecls/visa_multi/train.json
head -n 2 data/_onecls/visa_multi/test.json

# (optional) line counts
wc -l data/_onecls/visa_multi/train.json
wc -l data/_onecls/visa_multi/test.json
