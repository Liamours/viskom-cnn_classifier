"""
Download trashnet from HuggingFace and save raw images to data/raw/{class_name}/.
Skips if already downloaded.
"""

import os
from datasets import load_dataset
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def already_downloaded(label_names: list[str]) -> bool:
    """Check if all class folders exist and are non-empty."""
    for name in label_names:
        folder = RAW_DIR / name
        if not folder.exists() or len(list(folder.iterdir())) == 0:
            return False
    return True


def save_dataset(split, label_names: list[str]):
    for name in label_names:
        (RAW_DIR / name).mkdir(parents=True, exist_ok=True)

    counters = {i: 0 for i in range(len(label_names))}

    for item in split:
        lbl = item["label"]
        img = item["image"]
        folder = RAW_DIR / label_names[lbl]
        out_path = folder / f"{label_names[lbl]}_{counters[lbl]:04d}.jpg"
        img.save(out_path, "JPEG")
        counters[lbl] += 1

        if sum(counters.values()) % 100 == 0:
            print(f"  Saved {sum(counters.values())} images...", end="\r")

    print(f"\nDone. Total saved: {sum(counters.values())}")
    for lbl_idx, name in enumerate(label_names):
        print(f"  {name}: {counters[lbl_idx]} images")


if __name__ == "__main__":
    print("Loading dataset info...")
    dataset = load_dataset("garythung/trashnet")
    split = dataset["train"]

    label_names = split.features["label"].names
    print(f"Classes: {label_names}")

    if already_downloaded(label_names):
        print("[SKIP] Raw dataset already exists in data/raw/. No duplicates written.")
    else:
        print(f"Saving to {RAW_DIR} ...")
        save_dataset(split, label_names)
        print(f"\nRaw dataset saved to: {RAW_DIR}")