#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explainability for UniAS: activation-norm heatmap from the last conv block
Usage example (from repo root):
  python tools/explain_unias.py \
    --config ./configs/mvtec_ad_screw.yaml \
    --image /home/<you>/Downloads/data/mvtec_ad_resized/screw/test/manipulated_front/001.png \
    --ckpt ./experiments/mvtec_ad_screw/screw_toy/screw_toy_ckpt.pth_best.pth.tar \
    --out ./explain_out/screw_explain

If --ckpt is omitted, it runs with random/imagenet weights (still shows feature hotspots).
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import cv2
import math
import yaml
import torch
import argparse
import numpy as np
from easydict import EasyDict

# --- import your repo modules (paths relative to repo root) ---
from modeling.model import build_model
from utils.misc_helper import set_random_seed

def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    return cfg

def safe_load_state(ckpt_path, model, device):
    """
    Loads ckpt with key 'state_dict' if present. Ignores mismatched shapes.
    Safe on CPU/GPU.
    """
    if ckpt_path is None or ckpt_path == "":
        print("[info] no checkpoint provided, using current model weights.")
        return
    if not os.path.isfile(ckpt_path):
        print(f"[warn] checkpoint not found: {ckpt_path}")
        return

    print(f"[info] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)

    # remove keys with size mismatch
    to_pop = []
    for k, v in state.items():
        if k in model.state_dict():
            if model.state_dict()[k].shape != v.shape:
                print(f"[warn] size mismatch -> drop: {k} {tuple(v.shape)} -> {tuple(model.state_dict()[k].shape)}")
                to_pop.append(k)
    for k in to_pop:
        state.pop(k, None)

    miss, unexp = model.load_state_dict(state, strict=False)
    if miss:
        print(f"[info] missing keys: {len(miss)} (ok for heads) -> e.g. {miss[:5]}")
    if unexp:
        print(f"[info] unexpected keys in ckpt: {len(unexp)} -> e.g. {unexp[:5]}")

def preprocess_image(bgr_path, target_size, mean, std):
    """
    OpenCV BGR -> RGB, resize to target_size (H, W), normalize using dataset stats.
    Returns torch tensor (1,3,H,W) and a display rgb image in [0,1] for overlay.
    """
    img_bgr = cv2.imread(bgr_path, cv2.IMREAD_COLOR)
    assert img_bgr is not None, f"Cannot read image: {bgr_path}"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    th, tw = target_size
    img_rgb_resized = cv2.resize(img_rgb, (tw, th), interpolation=cv2.INTER_AREA)

    # keep a copy for visualization in [0,1]
    disp = (img_rgb_resized.astype(np.float32) / 255.0).clip(0, 1)

    # normalize
    mean = np.array(mean, dtype=np.float32).reshape(1,1,3)
    std  = np.array(std,  dtype=np.float32).reshape(1,1,3)
    norm = (disp - mean) / std

    tensor = torch.from_numpy(norm.transpose(2,0,1)).unsqueeze(0).float()
    return tensor, disp

def heatmap_from_activation(act):
    """
    act: tensor [C,H,W] from a conv layer. We use L2-norm across channels.
    Returns numpy heat [H,W] in [0,1]
    """
    with torch.no_grad():
        cam = torch.sqrt(torch.clamp((act ** 2).sum(dim=0), min=1e-12))
        cam -= cam.min()
        cam /= (cam.max() + 1e-12)
    return cam.detach().cpu().numpy()

def colorize_and_overlay(heat, disp_rgb, alpha=0.45):
    """
    heat in [0,1], disp_rgb in [0,1] (H,W,3). Returns BGR uint8 for saving.
    """
    h_uint8 = (heat * 255).astype(np.uint8)
    h_color = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)  # BGR
    base_bgr = (disp_rgb[..., ::-1] * 255.0).astype(np.uint8)  # to BGR
    blended = cv2.addWeighted(h_color, alpha, base_bgr, 1 - alpha, 0)
    return blended

def register_last_conv_hook(model, acts_container):
    """
    EfficientNet-B4 last conv block: model.backbone.efficientnet.features[-1]
    We store activation into acts_container["feat"] as [C,H,W].
    """
    # navigate to the last feature block
    try:
        target_layer = model.backbone.efficientnet.features[-1]
    except Exception:
        # fallback: try common attribute names used in EfficientNet wrappers
        target_layer = None
        for name in ["features", "blocks"]:
            if hasattr(model.backbone, name):
                obj = getattr(model.backbone, name)
                if hasattr(obj, "__getitem__"):
                    try:
                        target_layer = obj[-1]
                        break
                    except Exception:
                        pass
        assert target_layer is not None, "Could not locate EfficientNet last conv block."

    def hook_fn(module, inputs, output):
        # output shape: [B,C,H,W]
        acts_container["feat"] = output[0].detach()

    handle = target_layer.register_forward_hook(hook_fn)
    return handle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to UniAS YAML config")
    parser.add_argument("--image", required=True, help="Path to test image (BGR path ok)")
    parser.add_argument("--ckpt", default="", help="Optional checkpoint path (.pth.tar)")
    parser.add_argument("--out",  default="./explain_out", help="Output directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_random_seed(133, reproduce=False)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    # --- load config & build model like train_val.py ---
    cfg = load_config(args.config)
    model = build_model(cfg).to(device)
    model.eval()

    # load checkpoint (optional)
    safe_load_state(args.ckpt, model, device)

    # --- hook last conv layer ---
    acts = {}
    hook = register_last_conv_hook(model, acts)

    # --- preprocess image as dataset does ---
    H, W = cfg.DATASET.INPUT_SIZE
    mean = cfg.DATASET.PIXEL_MEAN
    std  = cfg.DATASET.PIXEL_STD
    x, disp = preprocess_image(args.image, (H, W), mean, std)
    x = x.to(device)

    # --- forward once (no DDP, no training) ---
    with torch.no_grad():
        _ = model({"image": x}, training=False)

    hook.remove()
    assert "feat" in acts, "Hook failed to capture activations."

    # --- make heatmap from activations ---
    heat = heatmap_from_activation(acts["feat"])     # [H',W']
    heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_CUBIC)  # upsample to input size
    overlay = colorize_and_overlay(heat, disp, alpha=0.45)

    # --- save results ---
    base = os.path.splitext(os.path.basename(args.image))[0]
    out_heat = os.path.join(args.out, f"{base}_featnorm_heat.png")
    out_overlay = os.path.join(args.out, f"{base}_featnorm_overlay.png")
    cv2.imwrite(out_heat, (heat * 255).astype(np.uint8))
    cv2.imwrite(out_overlay, overlay)

    print(f"[ok] saved:\n  {out_heat}\n  {out_overlay}")

if __name__ == "__main__":
    main()
