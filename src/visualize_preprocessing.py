"""
Two visualizations:
  1. Preprocessing before/after  — raw pixel vs normalized (runtime transform)
  2. Augmentation examples       — original image + N augmented variants

Outputs -> results/figures/preprocessing/

Usage:
    python src/visualize_preprocessing.py
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageFilter
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2
import torch

RAW_DIR  = Path("data/raw/train")
PROC_DIR = Path("data/processed/train")
RAW_CSV  = Path("data/raw/train.csv")
OUT_DIR  = Path("results/figures/preprocessing")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = [
    "basophil", "eosinophil", "erythroblast", "ig",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]

SEED = 42
random.seed(SEED)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_raw_pil(fname: str) -> Image.Image:
    return Image.open(RAW_DIR / fname).convert("RGB")


def apply_transform(pil_img: Image.Image, size: int = 224) -> torch.Tensor:
    tf = v2.Compose([
        v2.Resize((size, size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD),
    ])
    t = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1)
    return tf(t)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.permute(1, 2, 0).numpy()
    img = img * np.array(STD) + np.array(MEAN)
    return np.clip(img, 0, 1)


def cutout(img: Image.Image, fill: int = 128) -> Image.Image:
    arr  = np.array(img).copy()
    h, w = arr.shape[:2]
    ch   = int(h * random.uniform(0.05, 0.20))
    cw   = int(w * random.uniform(0.05, 0.20))
    y    = random.randint(0, h - ch)
    x    = random.randint(0, w - cw)
    arr[y:y+ch, x:x+cw] = fill
    return Image.fromarray(arr)


def augment(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.rotate(random.uniform(-15, 15),
                         resample=Image.BILINEAR, fillcolor=(128, 128, 128))
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.choice([1, 1.5, 2])))
    if random.random() < 0.5:
        arr   = np.array(img).astype(np.float32)
        alpha = random.uniform(0.8, 1.2)
        beta  = random.uniform(-20, 20)
        img   = Image.fromarray(np.clip(alpha * arr + beta, 0, 255).astype(np.uint8))
    if random.random() < 0.5:
        img = cutout(img)
    return img


# ── 1. Preprocessing before / after ─────────────────────────────────────────

def plot_preprocessing(n_per_class: int = 1):
    """Show raw vs normalized for one sample per class."""
    df = pd.read_csv(RAW_CSV)

    n_classes = len(CLASS_NAMES)
    n_cols    = n_per_class * 2          # raw | normalized pairs
    fig, axes = plt.subplots(n_classes, n_cols,
                             figsize=(n_cols * 2.4, n_classes * 2.6))

    fig.suptitle("Preprocessing: Before (Raw) vs After (Normalized)",
                 fontsize=13, fontweight="bold", y=1.01)

    for row, cls_name in enumerate(CLASS_NAMES):
        cls_df  = df[df["class_name"] == cls_name]
        samples = cls_df.sample(n=n_per_class, random_state=SEED)["filename"].tolist()

        for col_pair, fname in enumerate(samples):
            pil_img = load_raw_pil(fname)
            norm_t  = apply_transform(pil_img)
            norm_img = denormalize(norm_t)

            # raw
            ax_raw = axes[row][col_pair * 2]
            ax_raw.imshow(np.array(pil_img))
            ax_raw.axis("off")
            if row == 0:
                ax_raw.set_title("Raw", fontsize=9, fontweight="bold")
            if col_pair == 0:
                ax_raw.set_ylabel(cls_name, fontsize=8, rotation=90,
                                  labelpad=4, va="center")

            # normalized
            ax_norm = axes[row][col_pair * 2 + 1]
            ax_norm.imshow(norm_img)
            ax_norm.axis("off")
            if row == 0:
                ax_norm.set_title("Normalized", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = OUT_DIR / "preprocessing_before_after.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 2. Augmentation examples ─────────────────────────────────────────────────

def plot_augmentation(n_aug: int = 7):
    """
    For each class: show original + n_aug augmented variants.
    Grid: n_classes rows x (1 + n_aug) cols.
    """
    df = pd.read_csv(RAW_CSV)

    n_classes = len(CLASS_NAMES)
    n_cols    = 1 + n_aug
    fig, axes = plt.subplots(n_classes, n_cols,
                             figsize=(n_cols * 2.2, n_classes * 2.6))

    fig.suptitle(f"Augmentation Examples: Original + {n_aug} Variants per Class",
                 fontsize=13, fontweight="bold", y=1.01)

    for row, cls_name in enumerate(CLASS_NAMES):
        cls_df = df[df["class_name"] == cls_name]
        fname  = cls_df.sample(n=1, random_state=SEED)["filename"].iloc[0]
        orig   = load_raw_pil(fname)

        # original
        ax = axes[row][0]
        ax.imshow(np.array(orig))
        ax.axis("off")
        ax.set_ylabel(cls_name, fontsize=8, rotation=90, labelpad=4, va="center")
        if row == 0:
            ax.set_title("Original", fontsize=9, fontweight="bold")

        # augmented variants
        for aug_i in range(n_aug):
            random.seed(SEED + row * 100 + aug_i)
            aug_img = augment(orig.copy())
            ax = axes[row][1 + aug_i]
            ax.imshow(np.array(aug_img))
            ax.axis("off")
            if row == 0:
                ax.set_title(f"Aug {aug_i+1}", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = OUT_DIR / "augmentation_examples.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Plotting preprocessing before/after...")
    plot_preprocessing(n_per_class=1)

    print("Plotting augmentation examples...")
    plot_augmentation(n_aug=7)

    print(f"\nDone. Outputs → {OUT_DIR}")
