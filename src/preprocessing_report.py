"""
Generate a comprehensive preprocessing report for BloodMNIST.
Run AFTER preprocess.py has completed.
Saves report to report/preprocessing_report.md
"""

import json
import pandas as pd
from pathlib import Path
from datetime import date

RAW_DIR  = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR  = Path(__file__).parent.parent / "data" / "processed"
LOG_DIR  = Path(__file__).parent.parent / "results" / "logs"
RPT_DIR  = Path(__file__).parent.parent / "report"
SPLITS   = ["train", "val", "test"]

CLASS_NAMES = [
    "basophil", "eosinophil", "erythroblast", "ig",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_stats() -> dict:
    stats_path = LOG_DIR / "dataset_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    return {}


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count images per class from 'label' column."""
    counts = {name: 0 for name in CLASS_NAMES}
    for lbl in df["label"]:
        counts[CLASS_NAMES[int(lbl)]] += 1
    return counts


def aug_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count augmented images per class."""
    if "augmented" not in df.columns:
        return {name: 0 for name in CLASS_NAMES}
    aug_df = df[df["augmented"] == True]
    counts = {name: 0 for name in CLASS_NAMES}
    for lbl in aug_df["label"]:
        counts[CLASS_NAMES[int(lbl)]] += 1
    return counts


def generate():
    RPT_DIR.mkdir(parents=True, exist_ok=True)

    # load data
    raw_dfs  = {s: pd.read_csv(RAW_DIR / f"{s}.csv")  for s in SPLITS}
    proc_dfs = {s: pd.read_csv(OUT_DIR / f"{s}.csv") for s in SPLITS}
    stats    = load_stats()

    raw_train_counts  = class_counts(raw_dfs["train"])
    proc_train_counts = class_counts(proc_dfs["train"])
    augmented_counts  = aug_counts(proc_dfs["train"])

    lines = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines += [
        "# Preprocessing Report — BloodMNIST",
        f"Generated: {date.today()}",
        "",
        "---",
        "",
        "## 1. Dataset Overview",
        "",
        "| Property | Value |",
        "|---|---|",
        "| Dataset | BloodMNIST (MedMNIST v2) |",
        "| Source | HuggingFace: danjacobellis/bloodmnist_224 |",
        "| Modality | Blood Cell Microscope (RGB) |",
        "| Task | Multi-Class Classification |",
        "| Classes | 8 |",
        "| Image Size | 224 × 224 px |",
        "| License | CC BY 4.0 |",
        "",
        "### Classes",
        "",
        "| Label | Class |",
        "|---|---|",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"| {i} | {name} |")

    # ── Split counts ────────────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## 2. Split Counts",
        "",
        "| Split | Raw | Processed |",
        "|---|---|---|",
    ]
    for s in SPLITS:
        lines.append(f"| {s} | {len(raw_dfs[s]):,} | {len(proc_dfs[s]):,} |")
    total_raw  = sum(len(raw_dfs[s])  for s in SPLITS)
    total_proc = sum(len(proc_dfs[s]) for s in SPLITS)
    lines.append(f"| **Total** | **{total_raw:,}** | **{total_proc:,}** |")

    # ── Class distribution before/after augmentation ─────────────────────────
    lines += [
        "",
        "---",
        "",
        "## 3. Class Distribution (Train Split)",
        "",
        "| Class | Before Aug | +Augmented | After Aug |",
        "|---|---|---|---|",
    ]
    for name in CLASS_NAMES:
        before = raw_train_counts[name]
        added  = augmented_counts[name]
        after  = proc_train_counts[name]
        lines.append(f"| {name} | {before:,} | +{added:,} | {after:,} |")

    total_before = sum(raw_train_counts.values())
    total_added  = sum(augmented_counts.values())
    total_after  = sum(proc_train_counts.values())
    lines += [
        f"| **Total** | **{total_before:,}** | **+{total_added:,}** | **{total_after:,}** |",
        "",
        f"> Target per class: **2,000 images**",
    ]

    # ── Augmentation pipeline ────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## 4. Augmentation Pipeline",
        "",
        "Applied only to underrepresented training classes (count < 2,000).",
        "Val and test splits are copied unchanged.",
        "",
        "| Transform | Parameters | Probability |",
        "|---|---|---|",
        "| Horizontal Flip | — | 0.50 |",
        "| Rotation | ±15° (bilinear, fill=128) | 0.50 |",
        "| Gaussian Blur | radius ∈ {1, 1.5, 2} | 0.30 |",
        "| Brightness/Contrast | α ∈ [0.8, 1.2], β ∈ [−20, 20] | 0.50 |",
        "| Cutout | 5–20% patch, fill=128 (gray) | 0.50 |",
        "",
        "**Notes:**",
        "- Horizontal flip valid for blood cell microscopy (no anatomical orientation constraint).",
        "- Cutout fill value 128 chosen to match mid-gray, minimising distribution shift.",
        "- Labels preserved exactly from source image (single-class copy).",
        "- Augmented images sampled with replacement from class pool.",
    ]

    # ── Normalization ────────────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## 5. Normalization",
        "",
        "Normalization is applied **at runtime** (in `dataset.py` transforms), not on disk.",
        "Pixel values are first scaled to [0, 1], then normalized channel-wise.",
        "",
        "| Stats | R | G | B |",
        "|---|---|---|---|",
        f"| ImageNet Mean *(used)* | {IMAGENET_MEAN[0]} | {IMAGENET_MEAN[1]} | {IMAGENET_MEAN[2]} |",
        f"| ImageNet Std *(used)*  | {IMAGENET_STD[0]} | {IMAGENET_STD[1]} | {IMAGENET_STD[2]} |",
    ]

    if stats:
        bm_mean = stats.get("bloodmnist_mean", ["—", "—", "—"])
        bm_std  = stats.get("bloodmnist_std",  ["—", "—", "—"])
        lines += [
            f"| BloodMNIST Mean | {bm_mean[0]} | {bm_mean[1]} | {bm_mean[2]} |",
            f"| BloodMNIST Std  | {bm_std[0]} | {bm_std[1]} | {bm_std[2]} |",
        ]

    lines += [
        "",
        "**Rationale for ImageNet stats:**  ",
        "All three backbone models (MobileNetV4, EfficientNetV2-S, GhostNetV3) are pretrained on "
        "ImageNet-1k with ImageNet normalization. The first convolutional layers — which are frozen "
        "during training — learned feature detectors calibrated to this distribution. "
        "Using ImageNet stats ensures the input to frozen layers stays within the expected range, "
        "preventing distribution mismatch.",
    ]

    # ── Pipeline summary ─────────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## 6. Full Pipeline Summary",
        "",
        "```",
        "download_bloodmnist.py  → data/raw/{train,val,test}/  + {split}.csv",
        "eda_raw.py              → verify counts, hash duplicates, plot raw distributions",
        "eda_preprocessed.py     → compare before/after augmentation, plot processed distributions",
        "remove_duplicates.py    → remove exact-duplicate images, update CSVs",
        "preprocess.py           → copy val/test, augment train to TARGET=2000/class",
        "                          compute BloodMNIST mean/std → results/logs/dataset_stats.json",
        "```",
        "",
        "---",
        "",
        f"*Report generated by `src/preprocessing_report.py` on {date.today()}.*",
    ]

    out_path = RPT_DIR / "preprocessing_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {out_path}")


if __name__ == "__main__":
    generate()
