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
SPLITS   = ["train", "val", "test"]
TARGET   = 5000
WORKERS  = 8

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


# ── Augmentation ────────────────────────────────────────────────────────────

def augment(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))

    if random.random() < 0.3:
        radius = random.choice([1, 1.5, 2])
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    if random.random() < 0.5:
        arr   = np.array(img).astype(np.float32)
        alpha = random.uniform(0.8, 1.2)   # contrast
        beta  = random.uniform(-20, 20)    # brightness
        arr   = np.clip(alpha * arr + beta, 0, 255).astype(np.uint8)
        img   = Image.fromarray(arr)

    return img


# ── Helpers ──────────────────────────────────────────────────────────────────

def primary_label(active_labels: str) -> str:
    return active_labels.split("|")[0]


def build_class_pool(df: pd.DataFrame) -> dict[str, list[str]]:
    pool: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        lbl = primary_label(row["active_labels"])
        pool.setdefault(lbl, []).append(row["filename"])
    return pool


# ── Plan ─────────────────────────────────────────────────────────────────────

def plan(df_train: pd.DataFrame) -> dict[str, int]:
    pool    = build_class_pool(df_train)
    counts  = {lbl: len(fnames) for lbl, fnames in pool.items()}
    to_aug  = {lbl: max(0, TARGET - cnt) for lbl, cnt in counts.items()}

    header  = f"\n{'Class':<20} {'Current':>8} {'Target':>8} {'+Aug':>8} {'Expected':>9} {'x/img':>7}"
    print(header)
    print("-" * len(header))
    for lbl in sorted(counts, key=lambda l: counts[l]):
        cur      = counts[lbl]
        need     = to_aug[lbl]
        expected = cur + need
        x_per    = round(need / cur, 1) if need > 0 else 0
        marker   = " *" if need > 0 else ""
        print(f"{lbl:<20} {cur:>8} {TARGET:>8} {need:>8} {expected:>9} {x_per:>6}x{marker}")

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

def process_train(df_train: pd.DataFrame, to_aug: dict[str, int]):
    src_dir = RAW_DIR / "train"
    dst_dir = OUT_DIR / "train"
    dst_dir.mkdir(parents=True, exist_ok=True)

    pool = build_class_pool(df_train)

    # Copy originals
    def _copy(fname):
        src = src_dir / fname
        dst = dst_dir / fname
        if not dst.exists():
            shutil.copy2(src, dst)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(_copy, row["filename"]) for _, row in df_train.iterrows()]
        for _ in tqdm(as_completed(futs), total=len(futs), desc="Copy train originals"):
            pass

    # Generate augmented images
    new_rows = []
    aug_counter = 0

    for lbl, n_needed in to_aug.items():
        if n_needed == 0:
            continue

        source_files = pool[lbl]
        # sample with replacement if needed
        samples = [random.choice(source_files) for _ in range(n_needed)]
        ref_rows = df_train.set_index("filename")

        for i, fname in enumerate(tqdm(samples, desc=f"Aug {lbl}", leave=False)):
            src_path = src_dir / fname
            img      = Image.open(src_path).convert("RGB")
            aug_img  = augment(img)

            new_fname = f"aug_{lbl}_{aug_counter:06d}.png"
            aug_img.save(dst_dir / new_fname)

            orig_row = ref_rows.loc[fname]
            new_rows.append({
                "filename":     new_fname,
                "split":        "train",
                "active_labels": orig_row["active_labels"],
                "label_count":  orig_row["label_count"],
                "width":        aug_img.width,
                "height":       aug_img.height,
                "channels":     3,
                "mean":         round(np.array(aug_img).mean(), 4),
                "std":          round(np.array(aug_img).std(),  4),
                "augmented":    True,
                "source_file":  fname,
                **{col: orig_row[col]
                   for col in df_train.columns
                   if col not in ["filename","split","active_labels","label_count",
                                  "width","height","channels","mean","std"]},
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

    # Per-class verification
    print(f"\n{'Class':<20} {'Final count':>12}")
    print("-" * 34)
    pool_final = {}
    for _, row in df_all.iterrows():
        lbl = primary_label(row["active_labels"])
        pool_final[lbl] = pool_final.get(lbl, 0) + 1
    for lbl in sorted(pool_final, key=lambda l: pool_final[l]):
        print(f"  {lbl:<20} {pool_final[lbl]:>8}")


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

            print(f"\nDone. Processed data -> {OUT_DIR}")
