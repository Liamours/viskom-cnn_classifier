"""
For each wrong prediction: show the misclassified image on top,
then below it a GT class example and a predicted class example.
3 such blocks stacked vertically.

Usage:
    python src/visualize_errors.py configs/ghostnetv3.yml
    python src/visualize_errors.py configs/ghostnetv3.yml --seed 7
"""

import argparse
import random
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import BloodMNISTDataset, CLASS_NAMES
from models  import build_model

# LaTeX-style serif font, all black text
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Times New Roman", "Times", "Georgia"],
    "text.color":        "black",
    "axes.labelcolor":   "black",
    "xtick.color":       "black",
    "ytick.color":       "black",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])
N_EXAMPLES    = 3


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def denormalize(tensor):
    img = tensor.permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def run_inference(model, dataset, device):
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())
    return all_labels, all_preds


def add_border(ax, color, lw=3):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp    = cfg["experiment"]["name"]
    dcfg   = cfg["data"]

    test_ds = BloodMNISTDataset("test", dcfg["processed_dir"], dcfg["input_size"])

    ckpt_path = Path(cfg["output"]["weights_dir"]) / f"{exp}_best_auc.pth"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded: {ckpt_path.name}")

    print("Running inference...")
    all_labels, all_preds = run_inference(model, test_ds, device)

    class_indices = {c: [] for c in range(len(CLASS_NAMES))}
    for i, lbl in enumerate(all_labels):
        class_indices[lbl].append(i)

    wrong = [i for i in range(len(all_labels)) if all_preds[i] != all_labels[i]]
    print(f"Wrong: {len(wrong)} / {len(all_labels)}")

    if len(wrong) < N_EXAMPLES:
        print(f"Not enough wrong predictions (need {N_EXAMPLES}).")
        return

    chosen_wrongs = random.sample(wrong, N_EXAMPLES)

    # ── figure layout ─────────────────────────────────────────────────────────
    # each block: row0 = wrong image (spans 2 cols), row1 = [gt | pred]
    # compact: 3 blocks with tight spacing

    fig = plt.figure(figsize=(5, N_EXAMPLES * 3.6))

    outer = gridspec.GridSpec(
        N_EXAMPLES, 1, figure=fig,
        hspace=0.12,
    )

    for block_i, wrong_idx in enumerate(chosen_wrongs):
        gt_cls   = all_labels[wrong_idx]
        pred_cls = all_preds[wrong_idx]

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 2,
            subplot_spec=outer[block_i],
            hspace=0.08, wspace=0.06,
            height_ratios=[1.6, 1],
        )

        # ── top: wrong image (spans 2 cols) ──────────────────────────────────
        ax_top = fig.add_subplot(inner[0, :])
        ax_top.imshow(denormalize(test_ds[wrong_idx][0]))
        ax_top.axis("off")
        add_border(ax_top, "#e74c3c", lw=3)

        # label below top image via xlabel trick
        ax_top.set_title(
            f"GT: {CLASS_NAMES[gt_cls]}    |    Pred: {CLASS_NAMES[pred_cls]}",
            fontsize=9, color="black", pad=4,
        )
        # ── bottom left: GT example ───────────────────────────────────────────
        gt_pool    = [i for i in class_indices[gt_cls] if i != wrong_idx]
        gt_example = random.choice(gt_pool)
        ax_gt = fig.add_subplot(inner[1, 0])
        ax_gt.imshow(denormalize(test_ds[gt_example][0]))
        ax_gt.axis("off")
        add_border(ax_gt, "#2ecc71", lw=2)
        ax_gt.set_xlabel(CLASS_NAMES[gt_cls], fontsize=7,
                         labelpad=2, color="black")
        ax_gt.xaxis.set_label_position("bottom")

        # ── bottom right: predicted class example ─────────────────────────────
        pred_example = random.choice(class_indices[pred_cls])
        ax_pred = fig.add_subplot(inner[1, 1])
        ax_pred.imshow(denormalize(test_ds[pred_example][0]))
        ax_pred.axis("off")
        add_border(ax_pred, "#e74c3c", lw=2)
        ax_pred.set_xlabel(CLASS_NAMES[pred_cls], fontsize=7,
                           labelpad=2, color="black")
        ax_pred.xaxis.set_label_position("bottom")

    out_dir = Path("results") / "figures" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{exp}_error_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
