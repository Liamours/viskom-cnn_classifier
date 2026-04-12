# Preprocessing Report — ChestMNIST
Generated: 2026-04-12

---

## Overview

This report documents all preprocessing steps applied to the ChestMNIST dataset prior to model training. Steps are ordered chronologically and are fully reproducible via scripts in `src/`.

---

## 1. Dataset Acquisition

| Property | Value |
|---|---|
| Dataset | ChestMNIST (MedMNIST v2) |
| Source | MedMNIST API (`pip install medmnist`) |
| Modality | Chest X-Ray (grayscale, sourced from NIH ChestX-ray14) |
| Task | Multi-Label Binary Classification — 14 disease labels |
| Image Size | 224 × 224 px (MedMNIST+ large-size variant) |
| Channels | 1 (grayscale), converted to RGB at DataLoader level |
| License | CC BY 4.0 |
| Script | `src/download_chestmnist.py` |

Images saved flat per split (no label subfolders). Each split accompanied by a CSV with full metadata.

### CSV Schema

| Column | Description |
|---|---|
| `filename` | Image filename |
| `split` | train / val / test |
| `active_labels` | Pipe-separated active disease labels (e.g. `atelectasis\|effusion`) |
| `label_count` | Number of active labels (0 = no_finding) |
| `width`, `height`, `channels` | Image dimensions |
| `mean`, `std` | Per-image pixel statistics |
| `atelectasis` ... `pneumothorax` | 14 binary label columns |

---

## 2. Dataset Verification

Verified downloaded counts against official MedMNIST v2 benchmark figures.

| Split | Official | Downloaded | Match |
|---|---|---|---|
| train | 78,468 | 78,468 | Yes |
| val | 11,219 | 11,219 | Yes |
| test | 22,433 | 22,433 | Yes |
| **Total** | **112,120** | **112,120** | **Yes** |

Script: `src/eda.py` → `dataset_verification.txt`

---

## 3. Exploratory Data Analysis

Script: `src/eda.py`. Outputs saved to `results/figures/eda/`.

### 3.1 Label Distribution

ChestMNIST is heavily imbalanced. `no_finding` dominates all splits. Counts below are per primary label (first positive label per image).

| Class | Train | Val | Test |
|---|---|---|---|
| hernia | 144 | 41 | 42 |
| pneumonia | 978 | 133 | 242 |
| fibrosis | 1,157 | 166 | 362 |
| edema | 1,690 | 200 | 413 |
| emphysema | 1,799 | 208 | 508 |
| cardiomegaly | 1,950 | 240 | 582 |
| pleural | 2,278 | 372 | 734 |
| consolidation | 3,263 | 447 | 957 |
| pneumothorax | 3,703 | 504 | 1,089 |
| mass | 3,988 | 625 | 1,133 |
| nodule | 4,375 | 611 | 1,334 |
| atelectasis | 7,991 | 1,119 | 2,420 |
| effusion | 9,260 | 1,292 | 2,754 |
| infiltration | 13,910 | 2,016 | 3,936 |
| no_finding | 42,399 | 6,075 | 11,928 |

### 3.2 Labels per Image

Multi-label nature confirmed: images carry 0–8 simultaneous labels. Label count = 0 maps to `no_finding`.

### 3.3 Pixel Statistics

All images are grayscale X-rays. Per-image pixel mean clusters around 100–140 (uint8 scale), std around 40–70. Distribution consistent across splits.

### 3.4 Image Size & Channels

All images uniform: 224 × 224 × 1. No size variation — no resize needed.

---

## 4. Duplicate Detection & Removal

Script: `src/eda.py` (detection) → `src/remove_duplicates.py` (removal)

- **Method:** MD5 hash of raw image bytes
- **Scope:** Cross-split, all 112,120 images
- **Duplicate groups found:** 25
- **Files removed:** 26

### Distribution of Duplicates

| Type | Groups | Files removed |
|---|---|---|
| train ↔ train | 22 | 23 |
| train ↔ val | 2 | 2 |
| val ↔ val | 4 | 4 |
| val ↔ test | 1 | 1 |
| test ↔ test | 2 | 2 |

**Note:** Group 11 had 3 identical files — 2 deleted, 1 kept.

### Deduplication Strategy

Priority: **train > val > test**. Within same split: keep lower filename index. Cross-split: image retained in higher-priority split only.

### Dataset After Deduplication

| Split | Before | Removed | After |
|---|---|---|---|
| train | 78,468 | 16 | 78,452 |
| val | 11,219 | 7 | 11,212 |
| test | 22,433 | 3 | 22,430 |
| **Total** | **112,120** | **26** | **112,094** |

CSVs updated to reflect deletions. Duplicate report: `results/figures/eda/duplicates.txt`.
Full deletion log: `results/logs/dataset_change_report.md`.

---

## 5. Augmentation Plan

Script: `src/preprocess.py`

Augmentation applied to **train split only**. Val and test copied unchanged to preserve evaluation integrity.

### Strategy

- **Target:** 5,000 images per class (primary label basis)
- **Classes eligible:** all train classes below 5,000
- **Sampling:** with replacement from class image pool (random aug params each draw)
- **Non-uniform:** each transform applied independently with its own probability

### Multi-Label Consideration

ChestMNIST images carry multiple simultaneous disease labels. When an image is augmented, **all original labels are preserved unchanged** on the augmented copy — geometric and photometric transforms do not alter disease presence. Augmentation eligibility is determined by **primary label** (first positive label in the 14-label vector), which is a practical simplification. True per-label frequency balancing across all 14 labels simultaneously would require a combinatorial solver and is outside the scope of this work. The primary-label strategy is documented here for transparency and reproducibility.

### Augmentation Transforms (train only)

| Transform | Parameters | Probability |
|---|---|---|
| Horizontal Flip | — | 0.50 |
| Rotation | ±10° (bilinear, fill=128) | 0.50 |
| Gaussian Blur | radius ∈ {1.0, 1.5, 2.0} | 0.30 |
| Brightness/Contrast Jitter | contrast ×[0.8–1.2], brightness ±20 | 0.50 |

### Augmentation Targets

| Class | Original | +Augmented | Final |
|---|---|---|---|
| hernia | 144 | 4,856 | 5,000 |
| pneumonia | 978 | 4,022 | 5,000 |
| fibrosis | 1,157 | 3,843 | 5,000 |
| edema | 1,690 | 3,310 | 5,000 |
| emphysema | 1,799 | 3,201 | 5,000 |
| cardiomegaly | 1,950 | 3,050 | 5,000 |
| pleural | 2,278 | 2,722 | 5,000 |
| consolidation | 3,263 | 1,737 | 5,000 |
| pneumothorax | 3,703 | 1,297 | 5,000 |
| mass | 3,988 | 1,012 | 5,000 |
| nodule | 4,375 | 625 | 5,000 |
| atelectasis | 7,991 | 0 | 7,991 |
| effusion | 9,260 | 0 | 9,260 |
| infiltration | 13,910 | 0 | 13,910 |
| no_finding | 42,399 | 0 | 42,399 |

| Metric | Count |
|---|---|
| Original train images | 78,452 |
| Augmented images generated | 29,675 |
| Total processed train images | 108,127 |
| Val images (unchanged) | 11,212 |
| Test images (unchanged) | 22,430 |
| **Grand total (processed)** | **141,769** |

### Output Structure

```
data/processed/
├── train/       ← original + augmented images (flat)
├── val/         ← original only (flat)
├── test/        ← original only (flat)
├── train.csv    ← includes `augmented` (bool) + `source_file` columns
├── val.csv
└── test.csv
```

---

## 6. Normalization (Runtime — not saved to disk)

Normalization is applied at DataLoader level via `torchvision.transforms`. Not baked into saved images (PNG is uint8; saving normalized floats to disk causes precision loss).

All three models share identical pretrained normalization stats (verified via `timm.get_pretrained_cfg()`):

| Model | timm ID | Input Size | Mean | Std |
|---|---|---|---|---|
| MobileNetV3 | `mobilenetv3_large_100` | 3×224×224 | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) |
| EfficientNetV2 | `tf_efficientnetv2_b0` | 3×192×192 | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) |
| ConvNeXtV2 | `convnextv2_base.fcmae_ft_in22k_in1k` | 3×224×224 | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) |

Grayscale images converted to RGB at load time (`Image.convert("RGB")`).
EfficientNetV2 resized to 192×192; others to 224×224 — handled per-model in `dataset.py`.

---

## 7. Reproducibility

All steps fully reproducible in order:

```bash
python src/download_chestmnist.py   # acquire dataset
python src/eda.py                   # EDA + duplicate detection
python src/remove_duplicates.py     # remove duplicates, update CSVs
python src/dataset_report.py        # generate change report
python src/preprocess.py            # augment + copy to data/processed/
```
