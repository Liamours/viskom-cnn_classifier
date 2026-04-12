import csv
import numpy as np
from pathlib import Path
from PIL import Image
from medmnist import ChestMNIST
from medmnist.info import INFO
from tqdm import tqdm

RAW_DIR     = Path(__file__).parent.parent / "data" / "raw"
SIZE        = 224
SPLITS      = ["train", "val", "test"]
LABEL_NAMES = INFO["chestmnist"]["label"]
LABEL_COLS  = [LABEL_NAMES[str(i)] for i in range(len(LABEL_NAMES))]

CSV_FIELDS  = (
    ["filename", "split", "active_labels", "label_count",
     "width", "height", "channels", "mean", "std"]
    + LABEL_COLS
)


def save_split(split: str):
    split_dir = RAW_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)

    dataset = ChestMNIST(split=split, download=True, size=SIZE)
    rows = []

    for idx, (img_tensor, label_vec) in enumerate(tqdm(dataset, desc=f"[{split}]")):
        img  = Image.fromarray(np.array(img_tensor))
        arr  = np.array(img)
        vec  = np.array(label_vec).flatten().astype(int)

        fname        = f"{split}_{idx:05d}.png"
        active       = [LABEL_NAMES[str(i)] for i, v in enumerate(vec) if v == 1]
        active_str   = "|".join(active) if active else "no_finding"
        channels     = 1 if arr.ndim == 2 else arr.shape[2]

        img.save(split_dir / fname)

        row = {
            "filename":     fname,
            "split":        split,
            "active_labels": active_str,
            "label_count":  int(vec.sum()),
            "width":        img.width,
            "height":       img.height,
            "channels":     channels,
            "mean":         round(float(arr.mean()), 4),
            "std":          round(float(arr.std()),  4),
        }
        for col, val in zip(LABEL_COLS, vec):
            row[col] = int(val)

        rows.append(row)

    csv_path = RAW_DIR / f"{split}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {len(rows)} images + {csv_path.name}")


def already_done() -> bool:
    return all(
        (RAW_DIR / s).exists()
        and any((RAW_DIR / s).glob("*.png"))
        and (RAW_DIR / f"{s}.csv").exists()
        for s in SPLITS
    )


if __name__ == "__main__":
    if already_done():
        print("[SKIP] data/raw already populated with images + CSVs.")
    else:
        print(f"Downloading ChestMNIST {SIZE}x{SIZE} -> {RAW_DIR}\n")
        for split in SPLITS:
            save_split(split)
        print("\nDone.")
