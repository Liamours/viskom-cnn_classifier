import json
import shutil
import random
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

RAW_DIR  = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR  = Path(__file__).parent.parent / "data" / "processed"
LOG_DIR  = Path(__file__).parent.parent / "results" / "logs"
SPLITS   = ["train", "val", "test"]
TARGET   = 2000
WORKERS  = 8

# ImageNet stats — kept because frozen backbone layers expect this distribution
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Augmentation ────────────────────────────────────────────────────────────

def cutout(img: Image.Image,
           min_ratio: float = 0.05,
           max_ratio: float = 0.20,
           fill: int = 128) -> Image.Image:
    arr   = np.array(img).copy()
    h, w  = arr.shape[:2]
    cut_h = int(h * random.uniform(min_ratio, max_ratio))
    cut_w = int(w * random.uniform(min_ratio, max_ratio))
    y     = random.randint(0, h - cut_h)
    x     = random.randint(0, w - cut_w)
    arr[y:y + cut_h, x:x + cut_w] = fill
    return Image.fromarray(arr)


def augment(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))

    if random.random() < 0.3:
        radius = random.choice([1, 1.5, 2])
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    if random.random() < 0.5:
        arr   = np.array(img).astype(np.float32)
        alpha = random.uniform(0.8, 1.2)
        beta  = random.uniform(-20, 20)
        arr   = np.clip(alpha * arr + beta, 0, 255).astype(np.uint8)
        img   = Image.fromarray(arr)

    if random.random() < 0.5:
        img = cutout(img, fill=128)

    return img


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_class_pool(df: pd.DataFrame) -> dict[int, list[str]]:
    """Map label int → list of filenames."""
    pool: dict[int, list[str]] = {}
    for _, row in df.iterrows():
        lbl = int(row["label"])
        pool.setdefault(lbl, []).append(row["filename"])
    return pool


# ── Plan ─────────────────────────────────────────────────────────────────────

def plan(df_train: pd.DataFrame) -> dict[int, int]:
    from dataset import CLASS_NAMES
    pool    = build_class_pool(df_train)
    counts  = {lbl: len(fnames) for lbl, fnames in pool.items()}
    to_aug  = {lbl: max(0, TARGET - cnt) for lbl, cnt in counts.items()}

    header = f"\n{'Class':<20} {'Label':>6} {'Current':>8} {'Target':>8} {'+Aug':>8} {'Expected':>9}"
    print(header)
    print("-" * len(header))
    for lbl in sorted(counts):
        name     = CLASS_NAMES[lbl]
        cur      = counts[lbl]
        need     = to_aug[lbl]
        expected = cur + need
        marker   = " *" if need > 0 else ""
        print(f"{name:<20} {lbl:>6} {cur:>8} {TARGET:>8} {need:>8} {expected:>9}{marker}")

    total_orig = len(df_train)
    total_aug  = sum(to_aug.values())
    print(f"\nOriginal train images : {total_orig:,}")
    print(f"Augmented to generate : {total_aug:,}")
    print(f"Expected processed    : {total_orig + total_aug:,}")
    print(f"Target per class      : {TARGET:,}  (* = will be augmented)\n")

    return to_aug


# ── Copy unchanged splits ────────────────────────────────────────────────────

def copy_split(split: str):
    src_dir = RAW_DIR / split
    dst_dir = OUT_DIR / split
    dst_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DIR / f"{split}.csv")

    def _copy(fname):
        src = src_dir / fname
        dst = dst_dir / fname
        if not dst.exists():
            shutil.copy2(src, dst)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(_copy, row["filename"]) for _, row in df.iterrows()]
        for _ in tqdm(as_completed(futs), total=len(futs), desc=f"Copy {split}"):
            pass

    df.to_csv(OUT_DIR / f"{split}.csv", index=False)
    print(f"  {split}: {len(df)} images copied")


# ── Augment train ────────────────────────────────────────────────────────────

def process_train(df_train: pd.DataFrame, to_aug: dict[int, int]):
    src_dir = RAW_DIR / "train"
    dst_dir = OUT_DIR / "train"
    dst_dir.mkdir(parents=True, exist_ok=True)

    pool = build_class_pool(df_train)

    def _copy(fname):
        src = src_dir / fname
        dst = dst_dir / fname
        if not dst.exists():
            shutil.copy2(src, dst)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(_copy, row["filename"]) for _, row in df_train.iterrows()]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="Copy train originals"):
            pass

    new_rows  = []
    aug_counter = 0

    for lbl, n_needed in to_aug.items():
        if n_needed == 0:
            continue

        from dataset import CLASS_NAMES
        class_name   = CLASS_NAMES[lbl]
        source_files = pool[lbl]
        samples      = [random.choice(source_files) for _ in range(n_needed)]
        ref_rows     = df_train.set_index("filename")

        for fname in tqdm(samples, desc=f"Aug {class_name}", leave=False):
            src_path = src_dir / fname
            img      = Image.open(src_path).convert("RGB")
            aug_img  = augment(img)

            new_fname = f"aug_{class_name}_{aug_counter:06d}.png"
            aug_img.save(dst_dir / new_fname)

            orig_row = ref_rows.loc[fname]
            new_rows.append({
                "filename":   new_fname,
                "split":      "train",
                "class_name": orig_row["class_name"],
                "label":      int(orig_row["label"]),
                "width":      aug_img.width,
                "height":     aug_img.height,
                "channels":   3,
                "mean":       round(np.array(aug_img).mean(), 4),
                "std":        round(np.array(aug_img).std(),  4),
                "augmented":  True,
                "source_file": fname,
            })
            aug_counter += 1

    df_orig = df_train.copy()
    df_orig["augmented"]   = False
    df_orig["source_file"] = df_orig["filename"]

    df_aug = pd.DataFrame(new_rows)
    df_all = pd.concat([df_orig, df_aug], ignore_index=True)
    df_all.to_csv(OUT_DIR / "train.csv", index=False)

    print(f"\nTrain processed:")
    print(f"  Original : {len(df_orig):,}")
    print(f"  Augmented: {len(df_aug):,}")
    print(f"  Total    : {len(df_all):,}")

    from dataset import CLASS_NAMES
    print(f"\n{'Class':<20} {'Final count':>12}")
    print("-" * 34)
    pool_final: dict[str, int] = {}
    for _, row in df_all.iterrows():
        name = CLASS_NAMES[int(row["label"])]
        pool_final[name] = pool_final.get(name, 0) + 1
    for name in sorted(pool_final, key=lambda l: pool_final[l]):
        print(f"  {name:<20} {pool_final[name]:>8}")


# ── Dataset stats ────────────────────────────────────────────────────────────

def compute_dataset_stats(split: str = "train") -> dict:
    """Compute per-channel mean/std from original (non-augmented) processed images."""
    df       = pd.read_csv(OUT_DIR / f"{split}.csv")
    # only original images, not augmented
    if "augmented" in df.columns:
        df = df[df["augmented"] == False]

    img_dir  = OUT_DIR / split
    channels = 3
    n        = 0
    ch_sum   = np.zeros(channels, dtype=np.float64)
    ch_sq    = np.zeros(channels, dtype=np.float64)

    for fname in tqdm(df["filename"], desc="Computing stats"):
        arr = np.array(Image.open(img_dir / fname).convert("RGB")).astype(np.float64) / 255.0
        ch_sum += arr.mean(axis=(0, 1))
        ch_sq  += (arr ** 2).mean(axis=(0, 1))
        n += 1

    mean = (ch_sum / n).tolist()
    std  = np.sqrt(np.maximum(ch_sq / n - (ch_sum / n) ** 2, 0)).tolist()

    stats = {
        "split":         split,
        "n_images":      n,
        "bloodmnist_mean": [round(v, 4) for v in mean],
        "bloodmnist_std":  [round(v, 4) for v in std],
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std":  IMAGENET_STD,
        "used_for_training": "imagenet",
        "reason": (
            "Frozen backbone layers were pretrained with ImageNet normalization. "
            "Using ImageNet stats keeps input distribution consistent with what "
            "those layers expect."
        ),
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / "dataset_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n── Dataset Stats ({split}, {n} original images) ──")
    print(f"  BloodMNIST mean : {[round(v,4) for v in mean]}")
    print(f"  BloodMNIST std  : {[round(v,4) for v in std]}")
    print(f"  ImageNet mean   : {IMAGENET_MEAN}  ← used for training")
    print(f"  ImageNet std    : {IMAGENET_STD}  ← used for training")
    print(f"  Saved: {out_path}")

    return stats


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if (OUT_DIR / "train.csv").exists():
        print("[SKIP] data/processed already built.")
    else:
        df_train = pd.read_csv(RAW_DIR / "train.csv")

        print("=== Augmentation Plan ===")
        to_aug = plan(df_train)

        confirm = input("Proceed? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted.")
        else:
            print("\n=== Processing val + test (copy only) ===")
            copy_split("val")
            copy_split("test")

            print("\n=== Processing train (copy + augment) ===")
            process_train(df_train, to_aug)

            print("\n=== Computing Dataset Stats ===")
            compute_dataset_stats("train")

            print(f"\nDone. Processed data → {OUT_DIR}")
