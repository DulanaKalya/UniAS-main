import os, sys, io, yaml, cv2, torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from easydict import EasyDict as edict
from torchvision import transforms
from dotenv import load_dotenv

# ---------- Setup ----------
load_dotenv()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)

from modeling.model import UniAS  # UniAS expects a MODEL cfg (BACKBONE, MASK_FORMER, SEM_SEG_HEAD...)

# Optional Groq (for chat)
GROQ_OK = False
try:
    from groq import Groq
    if os.getenv("GROQ_API_KEY"):
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        GROQ_OK = True
except Exception:
    GROQ_OK = False

CFG_PATH = os.path.join(REPO_ROOT, "configs", "visa_part2_multi.yaml")
# <-- CHANGE THIS if your best ckpt has a different path/name
CKPT_PATH = os.path.join(REPO_ROOT, "experiments", "visa_part2_multi", "visa",  "visa_part2_multi_toy_ckpt.pth_best.pth.tar")

# ---------- Helpers ----------
def load_yaml_model_cfg(path: str) -> edict:
    with open(path, "r") as f:
        full = yaml.safe_load(f)
    return edict(full["MODEL"])

@st.cache_resource
def load_unias(ckpt_path: str):
    cfg_model = load_yaml_model_cfg(CFG_PATH)
    model = UniAS(cfg_model)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

# Same preprocessing as training
tx = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def _to_numpy_img(pil_img: Image.Image):
    arr = np.array(pil_img.convert("RGB"))
    return arr  # H,W,3 RGB

def _resize_like(src_hw, target_hw):
    return cv2.resize(src_hw, (target_hw.shape[1], target_hw.shape[0]), interpolation=cv2.INTER_CUBIC)

def _fuse_masks_to_heatmap(mask_tensor: torch.Tensor, target_hw) -> np.ndarray:
    """
    mask_tensor: (N, H, W) or (B, N, H, W) or (H, W)
    returns float32 heatmap in [0,1] resized to target image size
    """
    if mask_tensor.ndim == 2:
        hm = mask_tensor
    elif mask_tensor.ndim == 3:
        hm = mask_tensor.sigmoid().sum(dim=0)  # sum queries
    elif mask_tensor.ndim == 4:
        hm = mask_tensor.sigmoid().sum(dim=1).squeeze(0)  # B,N,H,W -> H,W (B=1)
    else:
        raise ValueError(f"Unexpected mask tensor shape: {mask_tensor.shape}")
    hm = hm.detach().cpu().float().numpy()
    hm = hm - hm.min()
    if hm.max() > 0:
        hm = hm / hm.max()
    hm = cv2.resize(hm, (target_hw.shape[1], target_hw.shape[0]), interpolation=cv2.INTER_CUBIC)
    return hm

def _feature_norm_heatmap(feat: torch.Tensor, target_hw) -> np.ndarray:
    """
    Fallback heatmap: L2 norm across channels of last feature, upsampled.
    feat: (B, C, H, W)
    """
    with torch.no_grad():
        nrm = torch.norm(feat.squeeze(0), dim=0)  # H,W
        nrm = nrm - nrm.min()
        if nrm.max() > 0:
            nrm = nrm / nrm.max()
        hm = nrm.cpu().numpy().astype(np.float32)
    hm = cv2.resize(hm, (target_hw.shape[1], target_hw.shape[0]), interpolation=cv2.INTER_CUBIC)
    return hm

def run_inference_and_heatmap(model: UniAS, pil_img: Image.Image, device="cpu"):
    rgb = _to_numpy_img(pil_img)  # H,W,3
    tens = tx(pil_img).unsqueeze(0).to(device)
    model = model.to(device)

    with torch.no_grad():
        out = model(tens, training=False)

    # Try common outputs first: pred_masks / masks
    heatmap = None
    # 1) MaskFormer-style
    for k in ["pred_masks", "masks"]:
        if isinstance(out, dict) and k in out:
            heatmap = _fuse_masks_to_heatmap(out[k], rgb)
            break
        if isinstance(out, (list, tuple)):
            for item in out:
                if isinstance(item, dict) and k in item:
                    heatmap = _fuse_masks_to_heatmap(item[k], rgb)
                    break
            if heatmap is not None:
                break

    # 2) Fallback: last feature map norm
    if heatmap is None:
        # Guess last feature map: try a few common names or tensors in tuple
        last_feat = None
        if isinstance(out, torch.Tensor) and out.ndim == 4:
            last_feat = out
        elif isinstance(out, (list, tuple)):
            # pick last tensor in the list/tuple
            cand = [x for x in out if isinstance(x, torch.Tensor) and x.ndim == 4]
            if len(cand) > 0:
                last_feat = cand[-1]
        elif isinstance(out, dict):
            # pick the largest 4D tensor in dict
            cand = [v for v in out.values() if isinstance(v, torch.Tensor) and v.ndim == 4]
            if len(cand) > 0:
                last_feat = max(cand, key=lambda t: t.shape[-1]*t.shape[-2])

        if last_feat is None:
            # As a last resort, make a flat heatmap (not ideal, but avoids crashing)
            heatmap = np.zeros(rgb.shape[:2], dtype=np.float32)
        else:
            heatmap = _feature_norm_heatmap(last_feat, rgb)

    # Simple scalar “anomaly score” as the 99th percentile of heatmap
    score = float(np.percentile(heatmap, 99) * 100.0)
    return rgb, heatmap, score

def make_compound(rgb: np.ndarray, heatmap: np.ndarray, bin_thresh: float = 0.5):
    """
    Returns a PIL image with 2x2: Input, Heatmap, Overlay, Binary
    """
    h, w = rgb.shape[:2]

    # Heatmap color
    hm8 = (heatmap * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm8, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = (0.6 * rgb + 0.4 * hm_color).clip(0, 255).astype(np.uint8)

    # Binary
    binary = (heatmap >= bin_thresh).astype(np.uint8) * 255
    binary_rgb = np.stack([binary, binary, binary], axis=-1)

    # Pack 2x2
    top = np.concatenate([rgb, hm_color], axis=1)
    bottom = np.concatenate([overlay, binary_rgb], axis=1)
    grid = np.concatenate([top, bottom], axis=0)
    return Image.fromarray(grid)

# ---------- UI ----------
st.set_page_config(page_title="PCB QC Assistant (UniAS + Groq)", layout="wide")
st.title("PCB QC Assistant – UniAS (pcb1/2/3/4)")

# Sidebar: model paths
with st.sidebar:
    st.subheader("Model")
    st.code(CKPT_PATH)
    device_opt = st.radio("Device", ["cpu", "cuda"], index=0 if not torch.cuda.is_available() else 1)

# Load model
try:
    model = load_unias(CKPT_PATH)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

col_l, col_r = st.columns([1, 1])

with col_l:
    uploaded = st.file_uploader("Upload a PCB image", type=["jpg", "jpeg", "png"])
    run_btn = st.button("Run Detection")

with col_r:
    st.markdown("**Config:**")
    st.code(CFG_PATH)

if uploaded and run_btn:
    try:
        pil = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Failed to read image: {e}")
        st.stop()

    rgb, heatmap, score = run_inference_and_heatmap(model, pil, device=device_opt)

    # Show compound panel
    comp = make_compound(rgb, heatmap, bin_thresh=0.5)
    st.subheader(f"Anomaly score (proxy): {score:.2f}%")
    st.image(comp, caption="Compound: [Input, Heatmap; Overlay, Binary]", use_column_width=True)

    # Download button
    buf = io.BytesIO()
    comp.save(buf, format="PNG")
    st.download_button("Download compound image", data=buf.getvalue(), file_name="compound.png", mime="image/png")

# --------- Conversational QC (Enter to submit) ----------
st.markdown("---")
st.header("Conversational QC")

if "chat" not in st.session_state:
    st.session_state.chat = []

# Show history
for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("Ask: Why is this PCB flagged? (press Enter)")
if prompt:
    st.session_state.chat.append(("user", prompt))
    # system message
    sys_msg = f"You are a QC assistant. An anomaly detector flagged a PCB image with a certain confidence. Explain likely reasons and recommended checks in practical terms for a technician."

    if GROQ_OK:
        try:
            resp = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt},
                ],
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            answer = f"(Groq error: {e})"
    else:
        answer = "Groq is not configured. Set GROQ_API_KEY in .env to enable AI explanations."

    st.session_state.chat.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)
