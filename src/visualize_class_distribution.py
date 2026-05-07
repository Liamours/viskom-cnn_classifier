"""
Clean class distribution figure across all splits.
Output -> results/figures/class_distribution.png

Usage:
    python src/visualize_class_distribution.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif", "Times New Roman", "Times", "Georgia"],
    "text.color":       "black",
    "axes.labelcolor":  "black",
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
})

RAW_DIR = Path("data/raw")
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]
CLASS_NAMES = [
    "basophil", "eosinophil", "erythroblast", "ig",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]
COLORS = ["#4e79a7", "#f28e2b", "#e15759"]


def count_classes(csv_path: Path) -> list[int]:
    df = pd.read_csv(csv_path)
    counts = df["class_name"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    return counts.tolist()


counts = {s: count_classes(RAW_DIR / f"{s}.csv") for s in SPLITS}

x     = np.arange(len(CLASS_NAMES))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))

for i, (split, color) in enumerate(zip(SPLITS, COLORS)):
    bars = ax.bar(x + i * width, counts[split], width,
                  label=split.capitalize(), color=color, edgecolor="white", linewidth=0.5)

ax.set_xticks(x + width)
ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right", fontsize=10)
ax.set_ylabel("Image Count", fontsize=11)
ax.set_xlabel("Class", fontsize=11)
ax.legend(fontsize=10, framealpha=0.9)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out = OUT_DIR / "class_distribution.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
