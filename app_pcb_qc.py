# app_pcb_qc.py
import os, io, base64, json
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F
from skimage.measure import label, regionprops

# ---- Groq (optional) ----
USE_GROQ = bool(os.environ.get("GROQ_API_KEY"))
if USE_GROQ:
    from groq import Groq
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# ---- UniAS imports ----
from modeling.model import UniAS
import yaml
from easydict import EasyDict as edict

# ---- EDIT THESE if your paths differ ----
MODEL_REGISTRY = {
    "pcb1": {
        "config": "./configs/visa_pcb1.yaml",
        "ckpt":   "./experiments/visa_pcb1/visa/visa_pcb1_ckpt.pth_best.pth.tar",
        "clsname": "pcb1"
    },
    "pcb2": {
        "config": "./configs/visa_pcb2.yaml",
        "ckpt":   "./experiments/visa_pcb2/visa/visa_pcb2_ckpt.pth_best.pth.tar",
        "clsname": "pcb2"
    },
    "pcb3": {
        "config": "./configs/visa_pcb3.yaml",
        "ckpt":   "./experiments/visa_pcb3/visa/visa_pcb3_ckpt.pth_best.pth.tar",
        "clsname": "pcb3"
    },
    "pcb4": {
        "config": "./configs/visa_pcb4.yaml",
        "ckpt":   "./experiments/visa_pcb4/visa/visa_pcb4_ckpt.pth_best.pth.tar",
        "clsname": "pcb4"
    },
}

def load_yaml(p): 
    with open(p, "r") as f: 
        return edict(yaml.load(f, Loader=yaml.FullLoader))

@st.cache_resource(show_spinner=False)
def load_model_for(config_path: str, ckpt_path: str):
    cfg = load_yaml(config_path)
    H, W = cfg.DATASET.INPUT_SIZE
    mean = cfg.DATASET.PIXEL_MEAN
    std  = cfg.DATASET.PIXEL_STD
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UniAS(cfg.MODEL).to(device).eval()

    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
        model.load_state_dict(state, strict=False)
    else:
        st.warning(f"Checkpoint not found: {ckpt_path}. Using backbone pretrained weights only.")

    return {"cfg": cfg, "model": model, "device": device,
            "H": H, "W": W, "mean": mean, "std": std}

def to_tensor(img_rgb: np.ndarray, size_hw: Tuple[int,int], mean, std) -> torch.Tensor:
    H, W = size_hw
    x = cv2.resize(img_rgb, (W, H)).astype(np.float32)/255.0
    x = (x - np.array(mean)) / np.array(std)
    return torch.from_numpy(x).permute(2,0,1).unsqueeze(0)

def normalize01(x: np.ndarray) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + 1e-6)

def colorize01(score01: np.ndarray) -> np.ndarray:
    arr = (score01 * 255).astype(np.uint8)
    cm = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
    return cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)

def post_process_regions(score01: np.ndarray, th: float=0.5, min_area: int=30):
    mask = (score01 > th).astype(np.uint8)
    lab = label(mask)
    props = regionprops(lab)
    H, W = mask.shape
    total = H * W
    regions = []
    for r in props:
        if r.area < min_area:
            continue
        y0, x0, y1, x1 = r.bbox
        sev = float(score01[r.coords[:,0], r.coords[:,1]].mean())
        regions.append({
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
            "area_px": int(r.area),
            "area_pct": 100.0 * r.area / total,
            "severity": sev
        })
    regions.sort(key=lambda z: z["severity"] * z["area_px"], reverse=True)
    return regions, mask

def quick_findings_for_pcb(regions, W: int):
    if not regions:
        return ["No region above threshold; likely normal."]
    lines = []
    for r in regions:
        x0,y0,x1,y1 = r["bbox"]
        cx = 0.5*(x0+x1)/W
        zone = int(np.clip(np.floor(cx*6)+1, 1, 6))  # split image width into 6 zones
        lines.append(f"Abnormal pattern near zone {zone}: area {r['area_pct']:.2f}%, severity {r['severity']:.2f}.")
    return lines

def encode_png(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok, "PNG encode failed"
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

def run_unias_inference(entry, img_rgb: np.ndarray, thr: float):
    model = entry["model"]; device = entry["device"]
    H, W = entry["H"], entry["W"]; mean, std = entry["mean"], entry["std"]
    x = to_tensor(img_rgb, (H, W), mean, std).to(device)

    with torch.no_grad():
        out = model({"image": x}, training=False)

    s = None
    if isinstance(out, dict):
        for k in ["pred_masks","anomaly_map","seg_score","score"]:
            if k in out: s = out[k]; break
    if s is None and isinstance(out,(list,tuple)) and len(out)>0:
        s = out[0]
    if s is None:
        s = out

    if s.dim()==4 and s.shape[1]>1:
        s = s.max(dim=1, keepdim=True)[0]
    if s.dim()==3:
        s = s.unsqueeze(1)

    s = F.interpolate(s, size=(H, W), mode="bilinear", align_corners=False)[0,0]
    s = s.float().detach().cpu().numpy()
    s01 = normalize01(s)

    regions, mask = post_process_regions(s01, th=thr)
    vis = cv2.resize(img_rgb, (W, H))
    jet = colorize01(s01)
    overlay = (0.45 * jet + 0.55 * vis).clip(0,255).astype(np.uint8)
    for r in regions[:10]:
        x0,y0,x1,y1 = r["bbox"]
        cv2.rectangle(overlay, (x0,y0), (x1,y1), (0,255,0), 2)
    findings = quick_findings_for_pcb(regions, W=W)
    overall = float(s01.mean()*0.3 + s01.max()*0.7)
    return {"overall": overall, "s01": s01, "regions": regions,
            "findings": findings, "overlay_png": encode_png(overlay)}

def groq_answer(clsname: str, overall: float, findings: List[str], question: str) -> str:
    if not USE_GROQ:
        base = f"[Offline mode] Class={clsname}, score={overall:.2f}\n"
        base += "No significant anomaly detected." if not findings else "Main findings:\n- " + "\n- ".join(findings)
        return base
    system = (
        "You are a Quality Control assistant for PCB visual inspection. "
        "Given the class, anomaly score, and short findings, answer the worker's question. "
        "Be concise, actionable, and mention likely cause + next step."
    )
    context = f"Class: {clsname}\nOverall score: {overall:.3f}\nFindings:\n- " + "\n- ".join(findings) + "\n"
    resp = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role":"system","content": system},
            {"role":"user","content": context + "\nQuestion: " + question}
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# --------- UI ---------
st.set_page_config(page_title="UniAS QC Assistant — VisA PCB", layout="wide")
st.title("🧭 UniAS QC Assistant — VisA PCB (pcb1–pcb4)")

left, right = st.columns([1.05, 1])
with left:
    pcb_choice = st.selectbox("Select PCB class / model:", list(MODEL_REGISTRY.keys()))
    thr = st.slider("Anomaly threshold", 0.00, 1.00, 0.50, 0.01)
    st.caption("Higher threshold → fewer regions, more confident anomalies.")
    up = st.file_uploader("Upload a PCB image (PNG/JPG)", type=["png","jpg","jpeg"])
    run_btn = st.button("Analyze")
with right:
    st.markdown("### Results")

if run_btn and up is not None:
    # ensure repo imports work
    os.environ.setdefault("PYTHONPATH", ".")
    entry = load_model_for(
        MODEL_REGISTRY[pcb_choice]["config"],
        MODEL_REGISTRY[pcb_choice]["ckpt"]
    )
    img = np.array(Image.open(io.BytesIO(up.getvalue())).convert("RGB"))
    out = run_unias_inference(entry, img, thr)

    c1, c2 = st.columns([1,1])
    with c1: st.image(up, caption="Input", use_column_width=True)
    with c2: st.image(out["overlay_png"], caption=f"Overlay (score={out['overall']:.2f})", use_column_width=True)

    st.markdown("**Findings:**")
    for f in out["findings"]:
        st.write("- " + f)

    st.markdown("---")
    st.subheader("Ask the QC Assistant")
    q = st.text_input("Question", "Why is this PCB flagged?")
    if st.button("Ask"):
        ans = groq_answer(
            clsname=MODEL_REGISTRY[pcb_choice]["clsname"],
            overall=out["overall"],
            findings=out["findings"],
            question=q
        )
        st.success(ans)
