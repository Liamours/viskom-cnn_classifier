"""
Load trashnet from HuggingFace and display 5 images per label.
"""

import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import defaultdict

# Load
print("Loading dataset...")
dataset = load_dataset("garythung/trashnet")
split = dataset["train"]

# Get label names
label_feature = split.features["label"]
label_names = label_feature.names
num_classes = len(label_names)
print(f"Classes ({num_classes}): {label_names}")
print(f"Total images: {len(split)}")

# Collect 5 samples per label
SAMPLES_PER_CLASS = 5
buckets = defaultdict(list)

for item in split:
    lbl = item["label"]
    if len(buckets[lbl]) < SAMPLES_PER_CLASS:
        buckets[lbl].append(item["image"])
    if all(len(v) >= SAMPLES_PER_CLASS for v in buckets.values()):
        break

# Plot
fig, axes = plt.subplots(num_classes, SAMPLES_PER_CLASS,
                         figsize=(SAMPLES_PER_CLASS * 3, num_classes * 3))
fig.suptitle("TrashNet — 5 samples per class", fontsize=14, y=1.01)

for row, lbl_idx in enumerate(sorted(buckets.keys())):
    for col, img in enumerate(buckets[lbl_idx]):
        ax = axes[row][col]
        ax.imshow(img)
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(label_names[lbl_idx], fontsize=11,
                          rotation=0, labelpad=60, va="center")

plt.tight_layout()
plt.savefig("results/figures/dataset_preview.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/figures/dataset_preview.png")