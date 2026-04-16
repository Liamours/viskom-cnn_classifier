import pandas as pd
from pathlib import Path
from datetime import date

RAW_DIR  = Path(__file__).parent.parent / "data" / "raw"
EDA_DIR  = Path(__file__).parent.parent / "results" / "figures" / "eda_raw"
OUT_DIR  = Path(__file__).parent.parent / "results" / "logs"
SPLITS   = ["train", "val", "test"]
PRIORITY = {s: i for i, s in enumerate(SPLITS)}

OFFICIAL = {"train": 11959, "val": 1712, "test": 3421, "total": 17092}

CLASS_NAMES = [
    "basophil", "eosinophil", "erythroblast", "ig",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]


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


def generate():
    dfs     = {s: pd.read_csv(RAW_DIR / f"{s}.csv") for s in SPLITS}
    dup_txt = EDA_DIR / "duplicates.txt"
    groups  = parse_duplicates(dup_txt)

    deletions: list[dict] = []
    for group in groups:
        keep, removes = resolve_keep(group)
        for r in removes:
            deletions.append({"keep": keep, "delete": r, "split": split_of(r)})

    lines = []
    lines += [
        f"# Dataset Change Report — BloodMNIST",
        f"Generated: {date.today()}",
        "",
        "---",
        "",
        "## 1. Source Dataset",
        "",
        "| Property | Value |",
        "|---|---|",
        "| Dataset | BloodMNIST (MedMNIST v2) |",
        "| Modality | Blood Cell Microscope (RGB) |",
        "| Task | Multi-Class Classification (8 classes) |",
        "| Image Size | 224 × 224 px |",
        "| Source | HuggingFace: danjacobellis/bloodmnist_224 |",
        "| License | CC BY 4.0 |",
        "",
        "---",
        "",
        "## 2. Classes",
        "",
        "| Label | Class Name |",
        "|---|---|",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"| {i} | {name} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Official Split Counts",
        "",
        "| Split | Official Count |",
        "|---|---|",
    ]
    for s in SPLITS:
        lines.append(f"| {s} | {OFFICIAL[s]:,} |")
    lines += [f"| **Total** | **{OFFICIAL['total']:,}** |", ""]

    lines += [
        "---",
        "",
        "## 4. Downloaded Count (Pre-Cleaning)",
        "",
        "| Split | Downloaded | Match Official |",
        "|---|---|---|",
    ]
    pre_totals = {}
    for s in SPLITS:
        n     = len(dfs[s])
        match = "Yes" if n == OFFICIAL[s] else f"No (delta: {n - OFFICIAL[s]:+d})"
        pre_totals[s] = n
        lines.append(f"| {s} | {n:,} | {match} |")
    pre_total = sum(pre_totals.values())
    lines += [f"| **Total** | **{pre_total:,}** | {'Yes' if pre_total == OFFICIAL['total'] else 'No'} |", ""]

    lines += [
        "---",
        "",
        "## 5. Duplicate Detection",
        "",
        f"- Method: MD5 hash of raw image bytes",
        f"- Scope: all splits cross-checked",
        f"- Duplicate groups found: **{len(groups)}**",
        f"- Total files to remove: **{len(deletions)}**",
        "",
        "### Deduplication Strategy",
        "",
        "Priority rule: **train > val > test**. Within same split: keep lower filename index.",
        "",
        "### Files Deleted",
        "",
        "| # | Deleted File | Split | Kept File |",
        "|---|---|---|---|",
    ]
    for i, group in enumerate(groups, 1):
        keep, removes = resolve_keep(group)
        for r in removes:
            lines.append(f"| {i} | `{r}` | {split_of(r)} | `{keep}` |")

    post = {s: pre_totals[s] - sum(1 for d in deletions if d["split"] == s) for s in SPLITS}
    post_total = sum(post.values())

    lines += [
        "",
        "---",
        "",
        "## 6. Dataset After Cleaning",
        "",
        "| Split | Before | Removed | After |",
        "|---|---|---|---|",
    ]
    for s in SPLITS:
        removed = sum(1 for d in deletions if d["split"] == s)
        lines.append(f"| {s} | {pre_totals[s]:,} | {removed} | {post[s]:,} |")
    lines += [
        f"| **Total** | **{pre_total:,}** | **{len(deletions)}** | **{post_total:,}** |",
        "",
        "---",
        "",
        "## 7. Notes",
        "",
        "- Removed images are exact pixel-level duplicates (identical MD5 hash).",
        "- Label metadata CSVs (train.csv, val.csv, test.csv) updated to reflect deletions.",
        "- No label re-assignment performed; only redundant entries removed.",
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "dataset_change_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {out_path}")


if __name__ == "__main__":
    generate()
