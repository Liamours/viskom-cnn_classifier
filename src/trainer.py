import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score


class Trainer:
    def __init__(self, model, optimizer, scheduler, cfg, device):
        self.model      = model
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.cfg        = cfg
        self.device     = device
        self.grad_clip  = cfg["training"].get("grad_clip")

        pos_weight = cfg["training"].get("pos_weight")
        pw = torch.tensor(pos_weight, dtype=torch.float32).to(device) \
            if pos_weight else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        self.best_score  = -float("inf")
        self.best_metric = cfg["output"]["save_best_metric"]
        self.weights_dir = Path(cfg["output"]["weights_dir"])
        self.exp_name    = cfg["experiment"]["name"]
        self.weights_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, loader) -> dict:
        self.model.train()
        total_loss  = 0.0
        all_preds   = []
        all_labels  = []

        for imgs, labels in tqdm(loader, desc="  train", leave=False):
            imgs, labels = (imgs.to(self.device, non_blocking=True),
                            labels.to(self.device, non_blocking=True))
            self.optimizer.zero_grad()
            logits = self.model(imgs)
            loss   = self.criterion(logits, labels)
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item() * imgs.size(0)

            preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.int().cpu().numpy())

        preds  = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        f1     = f1_score(labels, preds, average="macro", zero_division=0)
        ham    = (preds == labels).mean()

        return {
            "train_loss": total_loss / len(loader.dataset),
            "train_f1":   f1,
            "train_ham":  ham,
        }

    @torch.no_grad()
    def val_epoch(self, loader) -> dict:
        self.model.eval()
        total_loss  = 0.0
        all_logits  = []
        all_labels  = []

        for imgs, labels in tqdm(loader, desc="  val  ", leave=False):
            imgs, labels = (imgs.to(self.device, non_blocking=True),
                            labels.to(self.device, non_blocking=True))
            logits = self.model(imgs)
            loss   = self.criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        logits = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()
        probs  = 1 / (1 + np.exp(-logits))
        preds  = (probs >= 0.5).astype(int)

        try:
            auc = roc_auc_score(labels, probs, average="macro")
        except ValueError:
            auc = 0.0

        f1      = f1_score(labels, preds, average="macro", zero_division=0)
        ham     = (preds == labels.astype(int)).mean()
        subset  = (preds == labels.astype(int)).all(axis=1).mean()

        return {
            "val_loss":   total_loss / len(loader.dataset),
            "val_auc":    auc,
            "val_f1":     f1,
            "val_ham":    ham,
            "val_subset": subset,
        }

    def save_best(self, metrics: dict, epoch: int):
        score = metrics.get(self.best_metric, 0.0)
        if self.best_metric == "val_loss":
            score = -score

        if score > self.best_score:
            self.best_score = score
            path = self.weights_dir / f"{self.exp_name}_best.pth"
            torch.save(self.model.state_dict(), path)
            print(f"  [SAVED] best {self.best_metric}={score:.4f} → {path.name}")

    def step_scheduler(self, metrics: dict):
        if self.scheduler is None:
            return
        sched_name = self.cfg["training"]["scheduler"]
        if sched_name == "plateau":
            self.scheduler.step(metrics["val_loss"])
        else:
            self.scheduler.step()
