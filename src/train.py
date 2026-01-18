import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .dataset import HandwritingDataset
from .model import ViTClassifier, freeze_backbone
from .utils import (
    seed_everything, ensure_dir, load_json, save_json, save_yaml,
    cfg_to_dict, get_device
)
from .evaluate import evaluate_one_epoch

print(">>> train.py STARTED <<<")


def train(cfg: Config) -> Path:
    seed_everything(cfg.seed)
    device = get_device()

    run_dir = cfg.runs_dir / cfg.experiment_name
    ensure_dir(run_dir)

    save_yaml(cfg_to_dict(cfg), run_dir / "config.yaml")
    label_map = {name: i for i, name in enumerate(cfg.class_names)}
    save_json({"class_names": list(cfg.class_names), "label_map": label_map}, run_dir / "label.json")

    if not cfg.split_file.exists():
        raise FileNotFoundError(f"Split file not found: {cfg.split_file}. Run split first.")
    splits = load_json(cfg.split_file)

    train_ds = HandwritingDataset(splits["train"], img_size=cfg.img_size, train=True)
    val_ds = HandwritingDataset(splits["val"], img_size=cfg.img_size, train=False)

    '''
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    '''

    train_loader = DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=False
    )

    val_loader = DataLoader(
    val_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False
    )

    model = ViTClassifier(cfg.model_name, cfg.num_classes, dropout=cfg.dropout, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    best_val_acc = -1.0
    best_path = run_dir / "model.pt"
    metrics_log: Dict[str, Any] = {"history": []}

    def run_epochs(epochs: int, lr: float, phase_name: str):
        nonlocal best_val_acc

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=cfg.weight_decay
        )

        for epoch in range(1, epochs + 1):
            model.train()
            pbar = tqdm(train_loader, desc=f"{phase_name} | epoch {epoch}/{epochs}", leave=False)
            running_loss = 0.0
            correct = 0
            total = 0

            for x, y, _ in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == "cuda")):
                    logits = model(x)
                    loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * x.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += x.size(0)

                pbar.set_postfix(loss=loss.item(), acc=(correct / max(1, total)))

            train_loss = running_loss / max(1, total)
            train_acc = correct / max(1, total)

            # val
            val_metrics = evaluate_one_epoch(model, val_loader, device)
            val_acc = val_metrics["acc"]
            val_loss = val_metrics["loss"]

            entry = {
                "phase": phase_name,
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
            }
            metrics_log["history"].append(entry)

            # iyi olanı koru
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_name": cfg.model_name,
                        "class_names": list(cfg.class_names),
                        "img_size": cfg.img_size,
                    },
                    best_path
                )

            # her epoch için metrikleri tut
            save_json(metrics_log, run_dir / "metrics.json")

            print(
                f"[{phase_name}] epoch {epoch:02d} | "
                f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.3f} | best {best_val_acc:.3f}"
            )

    # 1) Head training
    freeze_backbone(model, freeze=True)
    run_epochs(cfg.epochs_head, cfg.lr_head, "head")

    # 2) Fine-tune all
    freeze_backbone(model, freeze=False)
    run_epochs(cfg.epochs_finetune, cfg.lr_finetune, "finetune")

    print(f"[OK] Best model saved at: {best_path}")
    return best_path


if __name__ == "__main__":
    out = train(Config())
    print(out)
