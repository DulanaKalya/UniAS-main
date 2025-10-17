# tools/peek_meta.py
import json, sys
from pathlib import Path

def load_json_any(p: Path):
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        items = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                items.append(json.loads(line))
        return items

def main(src_dir="data/MVTec-AD"):
    src = Path(src_dir)
    for split in ["train.json", "test.json"]:
        p = src / split
        print(f"\n=== {p} ===")
        items = load_json_any(p)
        print(f"count: {len(items)}")
        if not items:
            continue
        # show first 3 entries (keys + a guess at path fields)
        for i, m in enumerate(items[:3]):
            print(f"[{i}] keys:", list(m.keys()))
            for k in ["filename","image","img","img_path","path","file"]:
                if k in m:
                    print(f"    {k} -> {m[k]}")
    print("\nTip: find which key contains the image path (usually 'filename').")

if __name__ == "__main__":
    main(*sys.argv[1:])
