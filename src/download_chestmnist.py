import os
import numpy as np
from pathlib import Path
from PIL import Image
from medmnist import ChestMNIST
from medmnist.info import INFO

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
SIZE = 224
SPLITS = ["train", "val", "test"]

LABEL_NAMES = INFO["chestmnist"]["label"]
NO_FINDING = "no_finding"


def save_split(split: str):
    dataset = ChestMNIST(split=split, download=True, size=SIZE)
    out_base = RAW_DIR / split

    for lbl in list(LABEL_NAMES.values()) + [NO_FINDING]:
        (out_base / lbl).mkdir(parents=True, exist_ok=True)

    counter = {lbl: 0 for lbl in list(LABEL_NAMES.values()) + [NO_FINDING]}

    for idx, (img_tensor, label_vec) in enumerate(dataset):
        img = Image.fromarray(np.array(img_tensor))

        positive_indices = np.where(np.array(label_vec).flatten() == 1)[0]

        if len(positive_indices) == 0:
            folder_name = NO_FINDING
        else:
            folder_name = LABEL_NAMES[str(positive_indices[0])]

        count = counter[folder_name]
        out_path = out_base / folder_name / f"{folder_name}_{count:05d}.png"
        img.save(out_path)
        counter[folder_name] += 1

        if (idx + 1) % 500 == 0:
            print(f"  [{split}] {idx + 1}/{len(dataset)}", end="\r")

    print(f"\n[DONE] {split}: {sum(counter.values())} images")
    for lbl, cnt in counter.items():
        if cnt > 0:
            print(f"  {lbl}: {cnt}")


def already_done() -> bool:
    return all((RAW_DIR / s).exists() and any((RAW_DIR / s).iterdir()) for s in SPLITS)


if __name__ == "__main__":
    if already_done():
        print("[SKIP] data/raw already populated.")
    else:
        print(f"Saving ChestMNIST ({SIZE}x{SIZE}) to {RAW_DIR}\n")
        for split in SPLITS:
            save_split(split)
        print(f"\nAll splits saved to: {RAW_DIR}")
