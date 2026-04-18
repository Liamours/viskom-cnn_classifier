import sys
import yaml
import timm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image

BASE = Path(r"D:\Kuliah\Matkul\Semester 6\Visi Kompoter\Tugas CNN\github\viskom-cnn_classifier")
sys.path.insert(0, str(BASE / "src"))

from dataset  import BloodMNISTDataset, CLASS_NAMES
from evaluate import run_inference

log_dir = BASE / "results/logs"
out_dir = BASE / "results/figures/evaluation"
out_dir.mkdir(parents=True, exist_ok=True)

f1 = pd.read_csv(log_dir / "efficientnetv2/efficientnetv2_run1_log.csv")
f2 = pd.read_csv(log_dir / "mobilenetv4/mobilenetv4_run1_log.csv")
f3 = pd.read_csv(log_dir / "ghostnetv3/ghostnetv3_run1_log.csv")

for metric, ylabel, title, fname in [
    ("val_f1",   "Val F1",   "Validation F1",   "val_f1.jpg"),
    ("val_loss", "Val Loss", "Validation Loss",  "val_loss.jpg"),
]:
    plt.figure()
    for df, label in [(f1, "efficientnetv2"), (f2, "mobilenetv4"), (f3, "ghostnetv3")]:
        plt.plot(df["epoch"], df[metric], label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(out_dir / fname, format="jpg", dpi=300)
    plt.show()
    plt.close()

print("Validation curves saved.")

with open(BASE / "configs/ghostnetv3.yml") as f:
    cfg = yaml.safe_load(f)

cfg["data"]["processed_dir"]  = str(BASE / "data/processed")
cfg["output"]["weights_dir"]  = str(BASE / cfg["output"]["weights_dir"])

CKPT_TYPE = "best_auc"
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

dcfg    = cfg["data"]
test_ds = BloodMNISTDataset("test", dcfg["processed_dir"], dcfg["input_size"])
loader  = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

mcfg  = cfg["model"]
model = timm.create_model(mcfg["timm_id"], pretrained=False, num_classes=mcfg["num_classes"]).to(device)
ckpt  = Path(cfg["output"]["weights_dir"]) / f"{cfg['experiment']['name']}_{CKPT_TYPE}.pth"
model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
model.eval()
print(f"Loaded checkpoint: {ckpt.name}")

labels, probs, preds, infer_time = run_inference(model, loader, device)
correct_mask = preds == labels

print(f"Test: {len(test_ds):,} | Correct: {correct_mask.sum()} | Wrong: {(~correct_mask).sum()}")
print(f"Inference time: {infer_time:.3f}s")

all_labels = labels
all_preds  = preds

def load_raw(idx):
    """Load un-normalized PIL image from test dataset by index."""
    return Image.open(test_ds.root / test_ds.filenames[idx]).convert("RGB")

max_cols = 3

class_correct_pool = {c: [] for c in range(len(CLASS_NAMES))}
for idx in np.where(correct_mask)[0]:
    c = int(all_labels[idx])
    class_correct_pool[c].append(int(idx))

used_indices = set()

correct_sample = None
for c in range(len(CLASS_NAMES)):
    for idx in class_correct_pool[c]:
        if idx not in used_indices:
            correct_sample = (c, idx, c, True)
            used_indices.add(idx)
            break
    if correct_sample:
        break

wrong_samples = []
seen = set()
for idx in np.where(~correct_mask)[0]:
    tc = int(all_labels[idx])
    pc = int(all_preds[idx])
    if tc not in seen and idx not in used_indices:
        wrong_samples.append((tc, idx, pc, False))
        used_indices.add(idx)
        seen.add(tc)
    if len(wrong_samples) >= max_cols - 1:
        break

selected = []
if correct_sample:
    selected.append(correct_sample)
selected.extend(wrong_samples)
selected = selected[:max_cols]

n_cols = len(selected)
fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6), squeeze=False)

for col, (tc, idx, pc, is_correct) in enumerate(selected):
    ax_top = axes[0][col]
    ax_top.imshow(load_raw(idx), aspect="auto")
    ax_top.set_title(
        f"True: {CLASS_NAMES[tc]}\nPred: {CLASS_NAMES[pc]}",
        fontsize=9,
        color="green" if is_correct else "red",
        fontweight="bold",
    )
    ax_top.axis("off")

    ax_bot = axes[1][col]
    pool   = [p for p in class_correct_pool.get(pc, []) if p not in used_indices]

    if pool:
        ref_idx = pool[0]
        used_indices.add(ref_idx)
        ax_bot.imshow(load_raw(ref_idx), aspect="auto")
        ax_bot.set_title(
            f"Reference: {CLASS_NAMES[pc]}",
            fontsize=9,
            color="steelblue",
            fontweight="bold",
        )
    else:
        ax_bot.text(
            0.5, 0.5,
            f"No sample\n{CLASS_NAMES[pc]}",
            ha="center", va="center",
            transform=ax_bot.transAxes,
            fontsize=9,
        )
    ax_bot.axis("off")

plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.1)

save_path = str(out_dir / "ghostnetv3_combined.jpg")
plt.savefig(save_path, format="jpg", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

print(f"Saved: {save_path}")