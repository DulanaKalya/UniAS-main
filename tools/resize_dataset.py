# tools/resize_dataset.py
import os, argparse
from pathlib import Path
import cv2

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def is_mask_path(p: str) -> bool:
    # Treat ground-truth/mask files as masks; everything else as images
    parts = Path(p).parts
    name = Path(p).name.lower()
    return ("ground_truth" in parts) or name.endswith("_mask.png") or name.endswith("_gt.png")

def resize_write(src: Path, dst: Path, size: tuple[int,int]):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if is_mask_path(str(src)):
        im = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if im is None: 
            print(f"[warn] failed to read mask: {src}")
            return
        im = cv2.resize(im, size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(dst), im)
    else:
        im = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if im is None:
            print(f"[warn] failed to read image: {src}")
            return
        im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(dst), im)

def mirror_and_resize(src_root: Path, dst_root: Path, size=(224,224)):
    src_root = Path(src_root)
    count = 0
    for root, _, files in os.walk(src_root):
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in IMG_EXTS:
                s = Path(root) / f
                rel = s.relative_to(src_root)
                d = Path(dst_root) / rel
                resize_write(s, d, size)
                count += 1
                if count % 500 == 0:
                    print(f"[info] resized {count} files...")
    print(f"[done] total resized: {count}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source dataset root (original)")
    ap.add_argument("--dst", required=True, help="destination root (resized mirror)")
    ap.add_argument("--size", type=int, nargs=2, default=[224,224], help="width height")
    args = ap.parse_args()
    mirror_and_resize(Path(args.src), Path(args.dst), (args.size[0], args.size[1]))
