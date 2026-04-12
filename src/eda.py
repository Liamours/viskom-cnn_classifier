import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).parent.parent / "results" / "figures" / "eda"
SPLITS  = ["train", "val", "test"]
WORKERS = 8

EXPECTED = {"train": 78468, "val": 11219, "test": 22433, "total": 112120}

OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csvs() -> pd.DataFrame:
    frames = []
    for split in SPLITS:
        csv_path = RAW_DIR / f"{split}.csv"
        df = pd.read_csv(csv_path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def verify_dataset(df: pd.DataFrame):
    lines = ["=== Dataset Verification vs Official ChestMNIST ===\n"]
    lines.append(f"{'Split':<8} {'Expected':>10} {'Actual':>10} {'Match':>8}")
    lines.append("-" * 42)

    all_ok = True
    for split in SPLITS:
        expected = EXPECTED[split]
        actual   = len(df[df["split"] == split])
        match    = "OK" if actual == expected else "MISMATCH"
        if match == "MISMATCH":
            all_ok = False
        lines.append(f"{split:<8} {expected:>10} {actual:>10} {match:>8}")

    total_expected = EXPECTED["total"]
    total_actual   = len(df)
    total_match    = "OK" if total_actual == total_expected else "MISMATCH"
    if total_match == "MISMATCH":
        all_ok = False
    lines.append("-" * 42)
    lines.append(f"{'TOTAL':<8} {total_expected:>10} {total_actual:>10} {total_match:>8}")
    lines.append(f"\nResult: {'ALL COUNTS MATCH' if all_ok else 'DISCREPANCY FOUND'}")

    report = "\n".join(lines)
    print(report)
    (OUT_DIR / "dataset_verification.txt").write_text(report)


def hash_image(args: tuple) -> tuple[str, str]:
    img_path, fname = args
    return fname, hashlib.md5(Path(img_path).read_bytes()).hexdigest()


def find_duplicates(df: pd.DataFrame) -> dict[str, list[str]]:
    tasks = []
    for split in SPLITS:
        split_dir = RAW_DIR / split
        for fname in df[df["split"] == split]["filename"]:
            tasks.append((split_dir / fname, fname))

    hash_map = defaultdict(list)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(hash_image, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="Hashing images"):
            fname, h = fut.result()
            hash_map[h].append(fname)

    return {h: paths for h, paths in hash_map.items() if len(paths) > 1}


def report_duplicates(dupes: dict[str, list[str]]):
    path = OUT_DIR / "duplicates.txt"
    if not dupes:
        path.write_text("No duplicates found.\n")
        print("No duplicates found.")
    else:
        lines = [f"Found {len(dupes)} duplicate group(s):\n"]
        for h, paths in dupes.items():
            lines.append(f"\nHash: {h}")
            for p in paths:
                lines.append(f"  {p}")
        path.write_text("\n".join(lines))
        print(f"Duplicates: {len(dupes)} group(s) — see {path}")


def plot_label_distribution(df: pd.DataFrame):
    label_cols = [c for c in df.columns if c not in
                  ["filename", "split", "active_labels", "label_count",
                   "width", "height", "channels", "mean", "std"]]

    fig, axes = plt.subplots(1, len(SPLITS), figsize=(24, 8), sharey=False)
    fig.suptitle("Label Distribution per Split (multi-label counts)", fontsize=14)

    for ax, split in zip(axes, SPLITS):
        sub   = df[df["split"] == split][label_cols]
        counts = sub.sum().sort_values()
        counts.plot.barh(ax=ax)
        ax.set_title(split)
        ax.set_xlabel("Image count")
        for i, v in enumerate(counts):
            ax.text(v + 10, i, str(int(v)), va="center", fontsize=7)

    plt.tight_layout()
    path = OUT_DIR / "label_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_label_count_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(18, 5))
    fig.suptitle("Labels per Image Distribution (0 = no_finding)", fontsize=14)

    for ax, split in zip(axes, SPLITS):
        sub = df[df["split"] == split]["label_count"]
        unique, counts = np.unique(sub, return_counts=True)
        ax.bar([str(u) for u in unique], counts)
        ax.set_title(split)
        ax.set_xlabel("Number of active labels")
        ax.set_ylabel("Image count")
        for i, c in enumerate(counts):
            ax.text(i, c + 10, str(c), ha="center", fontsize=7)

    plt.tight_layout()
    path = OUT_DIR / "label_count_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_size_channel_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(3, len(SPLITS), figsize=(18, 12))
    fig.suptitle("Image Size & Channel Distribution", fontsize=14)
    metrics = [("width", "Width (px)"), ("height", "Height (px)"), ("channels", "Channels")]

    for col, split in enumerate(SPLITS):
        sub = df[df["split"] == split]
        for row, (key, label) in enumerate(metrics):
            vals = sub[key]
            unique, counts = np.unique(vals, return_counts=True)
            axes[row][col].bar([str(u) for u in unique], counts)
            axes[row][col].set_title(f"{split} — {label}")
            axes[row][col].set_xlabel(label)
            axes[row][col].set_ylabel("Count")

    plt.tight_layout()
    path = OUT_DIR / "size_channel_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_pixel_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(2, len(SPLITS), figsize=(18, 10))
    fig.suptitle("Pixel Value Distribution (per-image Mean & Std)", fontsize=14)

    for col, split in enumerate(SPLITS):
        sub = df[df["split"] == split]
        axes[0][col].hist(sub["mean"], bins=60, edgecolor="none")
        axes[0][col].set_title(f"{split} — Pixel Mean")
        axes[0][col].set_xlabel("Mean")
        axes[0][col].set_ylabel("Count")

        axes[1][col].hist(sub["std"], bins=60, edgecolor="none", color="orange")
        axes[1][col].set_title(f"{split} — Pixel Std")
        axes[1][col].set_xlabel("Std")
        axes[1][col].set_ylabel("Count")

    plt.tight_layout()
    path = OUT_DIR / "pixel_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Loading CSVs...")
    df = load_csvs()
    print(f"Total records: {len(df)}\n")

    print("Verifying dataset counts...")
    verify_dataset(df)

    print("\nChecking duplicates (hashing)...")
    dupes = find_duplicates(df)
    report_duplicates(dupes)

    print("\nPlotting EDA...")
    plot_label_distribution(df)
    plot_label_count_distribution(df)
    plot_pixel_distribution(df)

    print("\nDone. All outputs -> results/figures/eda/")
