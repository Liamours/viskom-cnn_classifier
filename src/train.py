import argparse
import csv
import random
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, ReduceLROnPlateau
)
from tqdm import tqdm

from dataset import BloodMNISTDataset
from models  import build_model
from trainer import Trainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, cfg: dict):
    tcfg = cfg["training"]
    params = filter(lambda p: p.requires_grad, model.parameters())
    name   = tcfg["optimizer"].lower()
    lr     = tcfg["lr"]
    wd     = tcfg.get("weight_decay", 1e-4)

    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=wd)
    elif name == "adam":
        return Adam(params, lr=lr, weight_decay=wd)
    elif name == "sgd":
        return SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, cfg: dict):
    tcfg   = cfg["training"]
    name   = tcfg["scheduler"].lower()
    epochs = tcfg["epochs"]
    warmup = tcfg.get("warmup_epochs", 0)

    if name == "none":
        return None
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs - warmup, eta_min=1e-6)
    if name == "step":
        return StepLR(optimizer, step_size=10, gamma=0.1)
    if name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    raise ValueError(f"Unknown scheduler: {name}")


def warmup_lr(optimizer, epoch: int, warmup_epochs: int, base_lr: float):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            pg["lr"] = lr


def save_log(log_rows: list, cfg: dict):
    logs_dir = Path(cfg["output"]["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"{cfg['experiment']['name']}_log.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Log saved: {path}")


def main(config_path: str):
    cfg    = load_config(config_path)
    set_seed(cfg["experiment"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Experiment : {cfg['experiment']['name']}\n")

    dcfg = cfg["data"]
    train_ds = BloodMNISTDataset("train", dcfg["processed_dir"], dcfg["input_size"])
    val_ds   = BloodMNISTDataset("val",   dcfg["processed_dir"], dcfg["input_size"])

    nw = dcfg["num_workers"]
    train_loader = DataLoader(
        train_ds, batch_size=dcfg["batch_size"], shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=dcfg["batch_size"] * 2, shuffle=False,
        num_workers=nw, pin_memory=True,
        persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None,
    )

    model     = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    trainer   = Trainer(model, optimizer, scheduler, cfg, device)

    # ── Pre-training warmup: a few batches to spin up workers ────────────────
    print("\nPreparing data loaders...")
    _iter = iter(train_loader)
    for _ in tqdm(range(min(4, len(train_loader))), desc="  warm-up"):
        imgs, labels = next(_iter)
        imgs.to(device, non_blocking=True)
    del _iter
    print("Ready.\n")

    tcfg      = cfg["training"]
    epochs    = tcfg["epochs"]
    warmup    = tcfg.get("warmup_epochs", 0)
    base_lr   = tcfg["lr"]
    log_rows  = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        warmup_lr(optimizer, epoch, warmup, base_lr)

        train_metrics = trainer.train_epoch(train_loader)
        val_metrics   = trainer.val_epoch(val_loader)

        if epoch >= warmup and scheduler is not None:
            trainer.step_scheduler(val_metrics)

        metrics = {**train_metrics, **val_metrics, "epoch": epoch + 1,
                   "lr": optimizer.param_groups[0]["lr"]}
        trainer.save_best(val_metrics)

        lr = optimizer.param_groups[0]['lr']
        print(f"  train_loss : {train_metrics['train_loss']:.4f}  |  train_acc : {train_metrics['train_acc']:.4f}  |  train_f1 : {train_metrics['train_f1']:.4f}")
        print(f"  val_loss   : {val_metrics['val_loss']:.4f}  |  val_acc   : {val_metrics['val_acc']:.4f}  |  val_f1   : {val_metrics['val_f1']:.4f}  (macro)")
        print(f"  val_auc    : {val_metrics['val_auc']:.4f}  |  lr        : {lr:.2e}\n")

        log_rows.append(metrics)

    save_log(log_rows, cfg)
    print("\nTraining complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
