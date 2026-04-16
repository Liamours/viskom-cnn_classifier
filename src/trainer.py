import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score


class Trainer:
    def __init__(self, model, optimizer, scheduler, cfg, device):
        self.model      = model
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.cfg        = cfg
        self.device     = device
        self.grad_clip  = cfg["training"].get("grad_clip")

        self.criterion   = nn.CrossEntropyLoss()

        self.weights_dir = Path(cfg["output"]["weights_dir"])
        self.exp_name    = cfg["experiment"]["name"]
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        self.best_auc  = -float("inf")
        self.best_loss =  float("inf")

    def train_epoch(self, loader) -> dict:
        self.model.train()
        total_loss = 0.0
        all_preds  = []
        all_labels = []

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

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

        preds  = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        acc    = (preds == labels).mean()
        f1     = f1_score(labels, preds, average="macro", zero_division=0)

        return {
            "train_loss": total_loss / len(loader.dataset),
            "train_acc":  acc,
            "train_f1":   f1,
        }

    @torch.no_grad()
    def val_epoch(self, loader) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []

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
        probs  = torch.softmax(torch.tensor(logits), dim=1).numpy()
        preds  = logits.argmax(axis=1)

        acc = (preds == labels).mean()
        f1  = f1_score(labels, preds, average="macro", zero_division=0)

        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        except ValueError:
            auc = 0.0

        return {
            "val_loss": total_loss / len(loader.dataset),
            "val_auc":  auc,
            "val_f1":   f1,
            "val_acc":  acc,
        }

    def save_best(self, metrics: dict):
        auc  = metrics.get("val_auc",  0.0)
        loss = metrics.get("val_loss", float("inf"))

        if auc > self.best_auc:
            self.best_auc = auc
            path = self.weights_dir / f"{self.exp_name}_best_auc.pth"
            torch.save(self.model.state_dict(), path)
            print(f"  [SAVED] best_auc={auc:.4f} → {path.name}")

        if loss < self.best_loss:
            self.best_loss = loss
            path = self.weights_dir / f"{self.exp_name}_best_loss.pth"
            torch.save(self.model.state_dict(), path)
            print(f"  [SAVED] best_loss={loss:.4f} → {path.name}")

        path = self.weights_dir / f"{self.exp_name}_latest.pth"
        torch.save(self.model.state_dict(), path)

    def step_scheduler(self, metrics: dict):
        if self.scheduler is None:
            return
        sched_name = self.cfg["training"]["scheduler"]
        if sched_name == "plateau":
            self.scheduler.step(metrics["val_loss"])
        else:
            self.scheduler.step()
