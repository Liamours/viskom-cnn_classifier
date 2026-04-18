import argparse
import time
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score
)

from dataset import BloodMNISTDataset, CLASS_NAMES
from models  import build_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_inference(model, loader, device):
    """Run forward pass — return (labels, probs, preds, inference_time_s)."""
    model.eval()
    all_logits, all_labels = [], []
    t0 = time.perf_counter()
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        all_logits.append(model(imgs).cpu())
        all_labels.append(labels)
    inference_time = time.perf_counter() - t0
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs  = torch.softmax(logits, dim=1).numpy()
    preds  = logits.argmax(dim=1).numpy()
    return labels, probs, preds, inference_time


@torch.no_grad()
def run_single_inference(model, sample_img: torch.Tensor, device) -> float:
    """Time inference for one sample. Return elapsed seconds."""
    model.eval()
    img = sample_img.unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model(img)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def save_classification_report(labels, preds, out_dir: Path, exp_name: str):
    report = classification_report(
        labels, preds,
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4,
    )
    print("\n── Classification Report ────────────────────────")
    print(report)
    path = out_dir / f"{exp_name}_classification_report.txt"
    path.write_text(report)
    print(f"Saved: {path}")


def save_confusion_matrix(labels, preds, out_dir: Path, exp_name: str):
    cm  = confusion_matrix(labels, preds)
    n   = len(CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"Confusion Matrix — {exp_name}", fontsize=12, fontweight="bold")

    thresh = cm.max() / 2
    for row in range(n):
        for col in range(n):
            ax.text(col, row, str(cm[row, col]),
                    ha="center", va="center", fontsize=8,
                    color="white" if cm[row, col] > thresh else "black")

    plt.tight_layout()
    path = out_dir / f"{exp_name}_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def save_metrics_summary(labels, probs, preds, out_dir: Path, exp_name: str,
                         inference_time_all: float, inference_time_one: float):
    acc       = (preds == labels).mean()
    f1        = f1_score(labels, preds, average="macro", zero_division=0)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall    = recall_score(labels, preds, average="macro", zero_division=0)
    n_samples = len(labels)

    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0

    lines = [
        f"=== Evaluation Summary — {exp_name} ===\n",
        f"AUC-ROC (macro OVR) : {auc:.4f}",
        f"F1 (macro)          : {f1:.4f}",
        f"Precision (macro)   : {precision:.4f}",
        f"Recall (macro)      : {recall:.4f}",
        f"Accuracy (top-1)    : {acc:.4f}",
        f"\nInference Time:",
        f"  Full dataset ({n_samples} samples) : {inference_time_all:.4f}s  ({inference_time_all/n_samples*1000:.3f} ms/sample)",
        f"  Single sample                    : {inference_time_one*1000:.3f} ms",
        f"\nPer-class AUC-ROC:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        try:
            per_auc = roc_auc_score(
                (labels == i).astype(int), probs[:, i]
            )
        except ValueError:
            per_auc = 0.0
        lines.append(f"  {name:<20} {per_auc:.4f}")

    summary = "\n".join(lines)
    print("\n── Metrics Summary ──────────────────────────────")
    print(summary)

    path = out_dir / f"{exp_name}_metrics_summary.txt"
    path.write_text(summary)
    print(f"Saved: {path}")


CHECKPOINTS = ["best_auc", "best_loss", "latest"]


def evaluate_checkpoint(model, ckpt_name: str, cfg: dict,
                        test_loader, device, out_dir: Path,
                        sample_img: torch.Tensor):
    exp       = cfg["experiment"]["name"]
    ckpt_file = f"{exp}_{ckpt_name}.pth"
    ckpt_path = Path(cfg["output"]["weights_dir"]) / ckpt_file

    if not ckpt_path.exists():
        print(f"  [SKIP] {ckpt_file} not found")
        return

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"\n{'='*52}")
    print(f"  Checkpoint : {ckpt_name}  ({ckpt_file})")
    print(f"{'='*52}")

    sub_dir = out_dir / ckpt_name
    sub_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{exp}_{ckpt_name}"

    labels, probs, preds, infer_all = run_inference(model, test_loader, device)
    infer_one = run_single_inference(model, sample_img, device)
    print(f"  Inference — full: {infer_all:.4f}s  |  single: {infer_one*1000:.3f}ms")
    save_metrics_summary(labels, probs, preds, sub_dir, tag, infer_all, infer_one)
    save_classification_report(labels, preds, sub_dir, tag)
    save_confusion_matrix(labels, preds, sub_dir, tag)


def main(config_path: str):
    cfg    = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp    = cfg["experiment"]["name"]

    out_dir = Path("results") / "evaluation" / exp
    out_dir.mkdir(parents=True, exist_ok=True)

    dcfg = cfg["data"]
    nw   = dcfg["num_workers"]

    test_ds = BloodMNISTDataset("test", dcfg["processed_dir"], dcfg["input_size"])
    test_loader = DataLoader(
        test_ds, batch_size=dcfg["batch_size"] * 2,
        shuffle=False, num_workers=nw, pin_memory=True,
        persistent_workers=nw > 0,
    )

    print(f"Device     : {device}")
    print(f"Experiment : {exp}")
    print(f"Test set   : {len(test_ds):,} images")

    model = build_model(cfg).to(device)

    # Fix sample index 0 — same image used across all checkpoints and all models
    sample_img, _ = test_ds[0]

    for ckpt_name in CHECKPOINTS:
        evaluate_checkpoint(model, ckpt_name, cfg, test_loader, device, out_dir, sample_img)

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    main(args.config)