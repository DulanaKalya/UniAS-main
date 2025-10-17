import importlib
import logging
import os
import random
import shutil
from collections.abc import Mapping
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist


# ------------------------------
# Logging utilities
# ------------------------------

def _noop_basicConfig(*args, **kwargs):
    """Mask global basicConfig to avoid duplicate root handlers from libraries."""
    return

# Prevent 3rd-party libs from calling logging.basicConfig and duplicating handlers
logging.basicConfig = _noop_basicConfig


def create_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create/get a logger that logs to both file and stdout without duplicating handlers.
    """
    log = logging.getLogger(name)
    log.setLevel(level)

    # Formatter
    fmt = logging.Formatter(
        "[%(filename)15s]%(message)s"  # compact format used by the repo
    )

    # Avoid duplicate handlers if called multiple times
    handler_keys = {type(h).__name__ + getattr(h, 'baseFilename', '') for h in log.handlers}

    # File handler
    if not os.path.isdir(os.path.dirname(os.path.abspath(log_file))):
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)

    fh_key = "FileHandler" + os.path.abspath(log_file)
    if fh_key not in handler_keys:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        fh.setLevel(level)
        log.addHandler(fh)

    # Stream handler
    if "StreamHandler" not in handler_keys:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(level)
        log.addHandler(sh)

    return log


def get_current_time() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ------------------------------
# Meters
# ------------------------------

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, length: int = 0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val: float, num: int = 1):
        if self.length > 0:
            # Keep a rolling window of values
            assert num == 1, "AverageMeter(length>0) expects num==1 updates."
            self.history.append(val)
            if len(self.history) > self.length:
                self.history.pop(0)
            self.val = self.history[-1]
            self.avg = float(np.mean(self.history)) if self.history else 0.0
        else:
            self.val = val
            self.sum += float(val) * int(num)
            self.count += int(num)
            self.avg = self.sum / max(self.count, 1)


# ------------------------------
# Checkpointing
# ------------------------------

def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def save_checkpoint(state: dict, is_best: bool, config, name: str = "ckpt.pth.tar"):
    """
    Save the latest checkpoint and (optionally) a best snapshot and per-epoch snapshots.
    - Ensures directory exists
    - Honors config.SAVER['ALWAYS_SAVE'] (defaults True if missing)
    """
    folder = getattr(config, "EXP_PATH", "./experiments/default")
    os.makedirs(folder, exist_ok=True)

    base_path = os.path.join(folder, name)
    torch.save(state, base_path)

    if is_best:
        shutil.copyfile(
            base_path,
            os.path.join(folder, f"{os.path.splitext(name)[0]}_best.pth.tar"),
        )

    always_save = True
    try:
        always_save = bool(config.SAVER.get("ALWAYS_SAVE", True))
    except Exception:
        pass

    if always_save and "epoch" in state:
        epoch = state["epoch"]
        shutil.copyfile(
            base_path,
            os.path.join(folder, f"{os.path.splitext(name)[0]}_{epoch}.pth.tar"),
        )


# ------------------------------
# Loading state
# ------------------------------

def load_state(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    """
    Robust checkpoint loader for single- or multi-GPU.
    - Works whether distributed is initialized or not
    - Prints info only on rank 0
    - Ignores size-mismatched keys gracefully (logs them)
    - If optimizer is provided, restores its state and returns (best_metric, epoch)
    - Returns (best_metric, epoch) if optimizer provided and load succeeds,
      else returns (0.0, -1) when not found / no optimizer.
    """
    rank = _get_rank()
    distributed = _is_dist()

    def _map_location(storage, loc):
        # Map tensors to the current CUDA device if available; else CPU
        if torch.cuda.is_available():
            return storage.cuda(torch.cuda.current_device())
        return storage.cpu()

    if not isinstance(path, str) or not os.path.isfile(path):
        if rank == 0:
            print(f"=> no checkpoint found at '{path}'")
        return (0.0, -1)

    if rank == 0:
        print(f"=> loading checkpoint '{path}'")

    checkpoint = torch.load(path, map_location=_map_location)

    # Handle common wrapping
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    model_dict = model.state_dict()

    # Filter out size-mismatch keys (warn once on rank 0)
    keep_keys = {}
    for k, v in state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            keep_keys[k] = v
        else:
            if rank == 0 and k in model_dict:
                print(f"caution: size-mismatch key: {k} size: {tuple(v.shape)} -> {tuple(model_dict[k].shape)}")
            # silently drop keys that don't exist in the current model or mismatch shape

    missing_keys = set(model_dict.keys()) - set(keep_keys.keys())
    unexpected_keys = set(state_dict.keys()) - set(keep_keys.keys())

    # Load with strict=False to allow partial load
    model.load_state_dict(keep_keys, strict=False)

    if rank == 0:
        for k in sorted(missing_keys):
            print(f"caution: missing key in checkpoint: {k}")
        # (No need to print unexpected_keys; they were filtered already.)

    best_metric = checkpoint.get("best_metric", 0.0)
    epoch = checkpoint.get("epoch", -1)

    if optimizer is not None and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if rank == 0:
                print(f"=> also loaded optimizer from checkpoint '{path}' (Epoch {epoch})")
        except Exception as e:
            if rank == 0:
                print(f"warning: failed to load optimizer state from '{path}': {e}")

    return (best_metric, epoch)


# ------------------------------
# Reproducibility
# ------------------------------

def set_random_seed(seed: int = 233, reproduce: bool = False):
    """
    Set seeds for Python, NumPy, and PyTorch. Optionally enable deterministic CUDNN.
    """
    np.random.seed(seed)
    torch.manual_seed(seed ** 2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed ** 3)
    random.seed(seed ** 4)

    if reproduce:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


# ------------------------------
# Device transfer
# ------------------------------

def to_device(batch: Mapping, device: str = "cuda", dtype=None) -> Mapping:
    """
    Recursively move a nested batch (dict/list/tuple/tensors) to a device/dtype.
    Special-case: if key 'image' exists, also apply dtype (common for fp16/bf16).
    """

    def move(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype if dtype is not None else x.dtype, non_blocking=True)
        elif isinstance(x, Mapping):
            return type(x)({k: move(v) for k, v in x.items()})
        elif isinstance(x, (list, tuple)):
            t = [move(v) for v in x]
            return type(x)(t) if not isinstance(x, tuple) else tuple(t)
        else:
            return x

    out = move(batch)

    # If there is a top-level 'image' tensor and dtype was provided but not applied above,
    # enforce dtype conversion explicitly (handles cases where image is not a Tensor yet).
    if "image" in out and torch.is_tensor(out["image"]):
        out["image"] = out["image"].to(device=device, dtype=dtype if dtype is not None else out["image"].dtype, non_blocking=True)

    return out
