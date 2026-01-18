import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

# Tekrarlanabilirlik
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# File & dir helpers
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_yaml(obj: Any, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


# Config helpers
def cfg_to_dict(cfg) -> Dict[str, Any]:
    out = {}
    for k, v in cfg.__dict__.items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out

# Device
def get_device():
    if torch.cuda.is_available():
        print("[INFO] Using CUDA")
        return torch.device("cuda")
    else:
        print("[INFO] Using CPU")
        return torch.device("cpu")
