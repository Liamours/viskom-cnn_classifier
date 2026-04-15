import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)

from dataset import ChestMNISTDataset, LABEL_COLS
from models  import build_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_inference(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1 / (1 + np.exp(-logits))
    preds  = (probs >= 0.5).astype(int)
    return labels, probs, preds


def save_classification_report(labels, preds, out_dir: Path, exp_name: str):
    report = classification_report(
        labels, preds,
        target_names=LABEL_COLS,
        zero_division=0,
        digits=4,
    )
    print("\n── Classification Report ────────────────────────")
    print(report)
    path = out_dir / f"{exp_name}_classification_report.txt"
    path.write_text(report)
    print(f"Saved: {path}")


def save_confusion_matrices(labels, preds, out_dir: Path, exp_name: str):
    n_labels = len(LABEL_COLS)
    cols     = 4
    rows     = (n_labels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2))
    axes      = axes.flatten()

    for i, label_name in enumerate(LABEL_COLS):
        cm  = confusion_matrix(labels[:, i], preds[:, i])
        ax  = axes[i]
        im  = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(label_name, fontsize=9, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"], fontsize=8)
        ax.set_yticklabels(["Neg", "Pos"], fontsize=8, rotation=90, va="center")
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)

        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(col, row, str(cm[row, col]),
                        ha="center", va="center",
                        fontsize=9, color="white" if cm[row, col] > cm.max() / 2 else "black")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Per-Label Confusion Matrices — {exp_name}", fontsize=12, y=1.01)
    plt.tight_layout()
    path = out_dir / f"{exp_name}_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def save_metrics_summary(labels, probs, preds, out_dir: Path, exp_name: str):
    try:
        auc = roc_auc_score(labels, probs, average="macro")
    except ValueError:
        auc = 0.0

    f1     = f1_score(labels, preds, average="macro", zero_division=0)
    ham    = (preds == labels.astype(int)).mean()
    subset = (preds == labels.astype(int)).all(axis=1).mean()

    lines = [
        f"=== Evaluation Summary — {exp_name} ===\n",
        f"AUC-ROC (macro)    : {auc:.4f}",
        f"F1 (macro)         : {f1:.4f}",
        f"Hamming accuracy   : {ham:.4f}",
        f"Subset accuracy    : {subset:.4f}",
        f"\nPer-label AUC-ROC:",
    ]
    for i, name in enumerate(LABEL_COLS):
        try:
            per_auc = roc_auc_score(labels[:, i], probs[:, i])
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
                         loader, device, out_dir: Path):
    exp      = cfg["experiment"]["name"]
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

    labels, probs, preds = run_inference(model, loader, device)
    save_metrics_summary(labels, probs, preds, sub_dir, tag)
    save_classification_report(labels, preds, sub_dir, tag)
    save_confusion_matrices(labels, preds, sub_dir, tag)


def main(config_path: str):
    cfg    = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp    = cfg["experiment"]["name"]

    out_dir = Path("results") / "evaluation" / exp
    out_dir.mkdir(parents=True, exist_ok=True)

    dcfg    = cfg["data"]
    test_ds = ChestMNISTDataset("test", dcfg["processed_dir"], dcfg["input_size"])
    loader  = DataLoader(
        test_ds, batch_size=dcfg["batch_size"] * 2,
        shuffle=False, num_workers=dcfg["num_workers"],
        pin_memory=True,
        persistent_workers=dcfg["num_workers"] > 0,
    )

    print(f"Device     : {device}")
    print(f"Experiment : {exp}")
    print(f"Test set   : {len(test_ds):,} images")

    model = build_model(cfg).to(device)

    for ckpt_name in CHECKPOINTS:
        evaluate_checkpoint(model, ckpt_name, cfg, loader, device, out_dir)

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    main(args.config)
