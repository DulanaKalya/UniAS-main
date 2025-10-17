# A) Create per-class metas
for C in pcb1 pcb2 pcb3 pcb4 pipe_fryum; do
  python tools/mvtec_filter_onecls.py \
    --src data/VisA \
    --out data/_onecls/visa_${C} \
    --cls ${C}
done

# B) Convert array JSON -> JSONL (what the loader expects)
for C in pcb1 pcb2 pcb3 pcb4 pipe_fryum; do
  python tools/json_array_to_jsonl.py data/_onecls/visa_${C}/train.json
  python tools/json_array_to_jsonl.py data/_onecls/visa_${C}/test.json
done

# C) Merge the five classes into one multi-class meta
mkdir -p data/_onecls/visa_part2_multi

cat data/_onecls/visa_pcb1/train.json \
    data/_onecls/visa_pcb2/train.json \
    data/_onecls/visa_pcb3/train.json \
    data/_onecls/visa_pcb4/train.json \
    data/_onecls/visa_pipe_fryum/train.json \
    > data/_onecls/visa_part2_multi/train.json

cat data/_onecls/visa_pcb1/test.json \
    data/_onecls/visa_pcb2/test.json \
    data/_onecls/visa_pcb3/test.json \
    data/_onecls/visa_pcb4/test.json \
    data/_onecls/visa_pipe_fryum/test.json \
    > data/_onecls/visa_part2_multi/test.json

# D) Sanity check (should start with '{', not '[')
head -n 2 data/_onecls/visa_part2_multi/train.json
head -n 2 data/_onecls/visa_part2_multi/test.json
