# tools/mvtec_filter_onecls.py
import argparse, json
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

def keep_record(m: dict, cls: str) -> bool:
    p = m.get("filename", "")
    s = p.lower().replace("\\", "/")
    cls = cls.lower()
    # accept: startswith "screw/", contains "/screw/", or endswith "/screw"
    return s.startswith(f"{cls}/") or (f"/{cls}/" in s) or s.endswith(f"/{cls}")

def run(src_dir: Path, out_dir: Path, cls: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train.json", "test.json"]:
        in_path = src_dir / split
        items = load_json_any(in_path)
        kept = [m for m in items if keep_record(m, cls)]
        (out_dir / split).write_text(json.dumps(kept, indent=2))
        print(f"[{split}] kept {len(kept)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/MVTec-AD")
    ap.add_argument("--out", default="data/_onecls/mvtec_screw")
    ap.add_argument("--cls", default="screw")
    args = ap.parse_args()
    run(Path(args.src), Path(args.out), args.cls)
