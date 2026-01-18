from pathlib import Path
import random

from config import Config
from utils import seed_everything, ensure_dir, save_json

def list_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]

def main():
    cfg = Config()
    seed_everything(cfg.seed)

    train, val = [], []

    for label, name in enumerate(cfg.class_names):
        class_dir = cfg.data_root / name
        files = list_images(class_dir)
        random.shuffle(files)

        n_val = max(1, int(len(files) * cfg.val_ratio))
        for p in files[n_val:]:
            train.append({"path": str(p), "label": label})
        for p in files[:n_val]:
            val.append({"path": str(p), "label": label})

    splits = {"train": train, "val": val}
    ensure_dir(cfg.runs_dir)
    save_json(splits, cfg.split_file)

    print(f"[OK] splits.json oluşturuldu | train={len(train)} val={len(val)}")

if __name__ == "__main__":
    main()
