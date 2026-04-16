"""
EDA on processed/augmented data (data/processed/).
Compares class distribution before vs after augmentation.
Outputs → results/figures/eda_preprocessed/
Run AFTER preprocess.py.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RAW_DIR  = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
OUT_DIR  = Path(__file__).parent.parent / "results" / "figures" / "eda_preprocessed"
SPLITS   = ["train", "val", "test"]

CLASS_NAMES = [
    "basophil", "eosinophil", "erythroblast", "ig",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]

OUT_DIR.mkdir(parents=True, exist_ok=True)


def class_counts(df: pd.DataFrame, augmented: bool | None = None) -> dict[str, int]:
    """Count images per class. augmented=None → all, True → aug only, False → original only."""
    if augmented is not None and "augmented" in df.columns:
        df = df[df["augmented"] == augmented]
    counts = {name: 0 for name in CLASS_NAMES}
    for lbl in df["label"]:
        counts[CLASS_NAMES[int(lbl)]] += 1
    return counts


def print_summary(raw_dfs: dict, proc_dfs: dict):
    print(f"\n{'Split':<8} {'Raw':>8} {'Processed':>12} {'Augmented':>12}")
    print("-" * 44)
    for split in SPLITS:
        raw_n  = len(raw_dfs[split])
        proc_n = len(proc_dfs[split])
        aug_n  = 0
        if "augmented" in proc_dfs[split].columns:
            aug_n = proc_dfs[split]["augmented"].sum()
        print(f"{split:<8} {raw_n:>8} {proc_n:>12} {aug_n:>12}")


def plot_class_comparison(raw_df: pd.DataFrame, proc_df: pd.DataFrame):
    """Stacked bar: original vs augmented per class (train only)."""
    orig_counts = class_counts(proc_df, augmented=False)
    aug_counts  = class_counts(proc_df, augmented=True)
    raw_counts  = class_counts(raw_df)

    x      = np.arange(len(CLASS_NAMES))
    width  = 0.35
    orig   = [orig_counts[c] for c in CLASS_NAMES]
    aug    = [aug_counts[c]  for c in CLASS_NAMES]
    raw    = [raw_counts[c]  for c in CLASS_NAMES]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Train Class Distribution — Before vs After Augmentation", fontsize=14)

    # Left: before vs after (grouped)
    ax = axes[0]
    ax.bar(x - width/2, raw,  width, label="Before (raw)",      color="steelblue")
    ax.bar(x + width/2, [o + a for o, a in zip(orig, aug)],
           width, label="After (processed)", color="darkorange")
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Image count"); ax.set_title("Before vs After")
    ax.legend()
    for i in range(len(CLASS_NAMES)):
        ax.text(i - width/2, raw[i] + 5,  str(raw[i]),  ha="center", fontsize=7)
        ax.text(i + width/2, orig[i] + aug[i] + 5, str(orig[i] + aug[i]), ha="center", fontsize=7)

    # Right: stacked original + augmented
    ax2 = axes[1]
    ax2.bar(x, orig, label="Original",  color="steelblue")
    ax2.bar(x, aug,  bottom=orig, label="Augmented", color="salmon")
    ax2.set_xticks(x); ax2.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Image count"); ax2.set_title("Processed (Original + Augmented)")
    ax2.legend()

    plt.tight_layout()
    path = OUT_DIR / "class_distribution_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_all_splits_distribution(proc_dfs: dict):
    """Class counts across all processed splits."""
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(18, 6), sharey=False)
    fig.suptitle("Class Distribution per Split — Processed", fontsize=14)

    for ax, split in zip(axes, SPLITS):
        counts = class_counts(proc_dfs[split])
        vals   = [counts[c] for c in CLASS_NAMES]
        ax.barh(CLASS_NAMES, vals)
        ax.set_title(split)
        ax.set_xlabel("Image count")
        for i, v in enumerate(vals):
            ax.text(v + 2, i, str(v), va="center", fontsize=8)

    plt.tight_layout()
    path = OUT_DIR / "class_distribution_all_splits.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_augmentation_gain(raw_df: pd.DataFrame, proc_df: pd.DataFrame):
    """Bar chart showing +N augmented per class."""
    orig_counts = class_counts(proc_df, augmented=False)
    aug_counts  = class_counts(proc_df, augmented=True)

    classes = CLASS_NAMES
    gains   = [aug_counts[c] for c in classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(classes, gains, color="salmon")
    ax.set_title("Augmented Images Added per Class (Train)", fontsize=13)
    ax.set_ylabel("Images added")
    ax.set_xlabel("Class")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    for bar, v in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 5, f"+{v}",
                ha="center", fontsize=9)

    plt.tight_layout()
    path = OUT_DIR / "augmentation_gain.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_pixel_distribution(proc_dfs: dict):
    fig, axes = plt.subplots(2, len(SPLITS), figsize=(18, 10))
    fig.suptitle("Pixel Value Distribution (per-image Mean & Std) — Processed", fontsize=14)

    for col, split in enumerate(SPLITS):
        df  = proc_dfs[split]
        # use original images only for fair comparison
        sub = df[df["augmented"] == False] if "augmented" in df.columns else df

        axes[0][col].hist(sub["mean"], bins=60, edgecolor="none")
        axes[0][col].set_title(f"{split} — Pixel Mean")
        axes[0][col].set_xlabel("Mean"); axes[0][col].set_ylabel("Count")

        axes[1][col].hist(sub["std"], bins=60, edgecolor="none", color="orange")
        axes[1][col].set_title(f"{split} — Pixel Std")
        axes[1][col].set_xlabel("Std"); axes[1][col].set_ylabel("Count")

    plt.tight_layout()
    path = OUT_DIR / "pixel_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Loading CSVs...")
    raw_dfs  = {s: pd.read_csv(RAW_DIR  / f"{s}.csv") for s in SPLITS}
    proc_dfs = {s: pd.read_csv(PROC_DIR / f"{s}.csv") for s in SPLITS}

    print("\n── Split Summary ────────────────────────────────")
    print_summary(raw_dfs, proc_dfs)

    print("\nPlotting EDA (processed)...")
    plot_class_comparison(raw_dfs["train"], proc_dfs["train"])
    plot_all_splits_distribution(proc_dfs)
    plot_augmentation_gain(raw_dfs["train"], proc_dfs["train"])
    plot_pixel_distribution(proc_dfs)

    print(f"\nDone. All outputs → {OUT_DIR}")
