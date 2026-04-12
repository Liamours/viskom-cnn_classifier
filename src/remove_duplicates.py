import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
EDA_DIR = Path(__file__).parent.parent / "results" / "figures" / "eda"
SPLITS  = ["train", "val", "test"]
PRIORITY = {s: i for i, s in enumerate(SPLITS)}


def parse_duplicates(txt_path: Path) -> list[list[str]]:
    groups = []
    current = []
    for line in txt_path.read_text().splitlines():
        if line.startswith("Hash:"):
            if current:
                groups.append(current)
            current = []
        elif line.strip() and not line.startswith("Found"):
            current.append(line.strip())
    if current:
        groups.append(current)
    return groups


def split_of(fname: str) -> str:
    return fname.split("_")[0]


def resolve_keep(group: list[str]) -> tuple[str, list[str]]:
    sorted_group = sorted(group, key=lambda f: (PRIORITY[split_of(f)], f))
    return sorted_group[0], sorted_group[1:]


def remove_duplicates():
    dup_path = EDA_DIR / "duplicates.txt"
    if not dup_path.exists():
        print("duplicates.txt not found. Run eda.py first.")
        return

    groups = parse_duplicates(dup_path)
    if not groups:
        print("No duplicates to remove.")
        return

    to_delete: list[tuple[str, str]] = []
    print(f"Resolving {len(groups)} duplicate group(s)...\n")
    for group in groups:
        keep, remove = resolve_keep(group)
        print(f"  KEEP   : {keep}")
        for r in remove:
            print(f"  DELETE : {r}")
            to_delete.append((split_of(r), r))
        print()

    dfs = {split: pd.read_csv(RAW_DIR / f"{split}.csv") for split in SPLITS}

    deleted = 0
    for split, fname in tqdm(to_delete, desc="Deleting files"):
        img_path = RAW_DIR / split / fname
        if img_path.exists():
            img_path.unlink()
            deleted += 1
        dfs[split] = dfs[split][dfs[split]["filename"] != fname]

    for split, df in dfs.items():
        df.to_csv(RAW_DIR / f"{split}.csv", index=False)
        print(f"Updated {split}.csv -> {len(df)} records")

    dup_path.write_text("No duplicates found.\n")
    print(f"\nDeleted {deleted} duplicate files. CSVs updated.")


if __name__ == "__main__":
    remove_duplicates()
