import json, sys
from pathlib import Path

p = Path(sys.argv[1])  # e.g., data/_onecls/mvtec_screw/train.json
data = json.loads(p.read_text())
assert isinstance(data, list), "Expected a JSON array"

with p.open("w") as f:
    for obj in data:
        f.write(json.dumps(obj) + "\n")
print(f"[ok] rewrote {p} as JSONL with {len(data)} lines")
