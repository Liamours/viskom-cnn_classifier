"""
Download pretrained weights for all 3 CNN architectures via timm.
Saves state_dicts to models/weights/.
"""

import os
import timm
import torch

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

MODELS = {
    "mobilenetv3":  "mobilenetv3_large_100",
    "convnextv2":   "convnextv2_base.fcmae_ft_in22k_in1k",
    "efficientnetv2": "tf_efficientnetv2_b0",
}

NUM_CLASSES = 5  # change to match your dataset


def download_and_save(name: str, timm_id: str, num_classes: int):
    out_path = os.path.join(WEIGHTS_DIR, f"{name}.pth")
    if os.path.exists(out_path):
        print(f"[SKIP] {name} already exists at {out_path}")
        return

    print(f"[DOWNLOAD] {name} ({timm_id}) ...")
    model = timm.create_model(timm_id, pretrained=True, num_classes=num_classes)
    model.eval()
    torch.save(model.state_dict(), out_path)
    print(f"[SAVED]    {out_path}")


def print_model_info(name: str, timm_id: str, num_classes: int):
    print(f"\n{'='*50}")
    print(f"Architecture : {name}")
    print(f"timm ID      : {timm_id}")
    model = timm.create_model(timm_id, pretrained=False, num_classes=num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg = model.default_cfg
    print(f"Input size   : {cfg.get('input_size', 'N/A')}")
    print(f"Total params : {total_params:,}")
    print(f"Trainable    : {trainable:,}")
    print(f"{'='*50}")


if __name__ == "__main__":
    print("=== Downloading pretrained weights ===\n")
    for name, timm_id in MODELS.items():
        download_and_save(name, timm_id, NUM_CLASSES)

    print("\n=== Model Architecture Info ===")
    for name, timm_id in MODELS.items():
        print_model_info(name, timm_id, NUM_CLASSES)

    print("\nDone. Weights saved to models/weights/")