"""
Download BloodMNIST 224x224 from HuggingFace (hf-mirror) and save as PNG + CSV.
Dataset: danjacobellis/bloodmnist_224
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import csv
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
SPLITS  = ["train", "val", "test"]

CLASS_NAMES = [
    "basophil", "eosinophil", "erythroblast", "ig",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]

# HuggingFace uses "validation" — map to "val"
HF_SPLIT_MAP = {"train": "train", "val": "validation", "test": "test"}

CSV_FIELDS = ["filename", "split", "class_name", "label",
              "width", "height", "channels", "mean", "std"]


def save_split(split: str, hf_data):
    split_dir = RAW_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, sample in enumerate(tqdm(hf_data, desc=f"[{split}]")):
        img   = sample["image"]
        lbl   = sample["label"]
        label = int(lbl[0]) if isinstance(lbl, (list, tuple)) else int(lbl)

        # ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        arr      = np.array(img)
        fname    = f"{split}_{idx:05d}.png"
        img.save(split_dir / fname)

        rows.append({
            "filename":   fname,
            "split":      split,
            "class_name": CLASS_NAMES[label],
            "label":      label,
            "width":      img.width,
            "height":     img.height,
            "channels":   3,
            "mean":       round(float(arr.mean()), 4),
            "std":        round(float(arr.std()),  4),
        })

    csv_path = RAW_DIR / f"{split}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {split}: {len(rows):,} images saved → {csv_path.name}")


def already_done() -> bool:
    return all(
        (RAW_DIR / s).exists()
        and any((RAW_DIR / s).glob("*.png"))
        and (RAW_DIR / f"{s}.csv").exists()
        for s in SPLITS
    )


if __name__ == "__main__":
    if already_done():
        print("[SKIP] data/raw already populated.")
    else:
        print("Loading BloodMNIST 224 from HuggingFace...\n")
        ds = load_dataset("danjacobellis/bloodmnist_224")

        for split in SPLITS:
            hf_split = HF_SPLIT_MAP[split]
            save_split(split, ds[hf_split])

        print("\nDone. Raw data → data/raw/")
