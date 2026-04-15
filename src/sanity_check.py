import argparse
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import ChestMNISTDataset, LABEL_COLS
from models  import build_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def check_data(cfg: dict, device: torch.device):
    print("── [1] Data Loading ─────────────────────────────")
    dcfg = cfg["data"]

    for split in ["train", "val", "test"]:
        ds = ChestMNISTDataset(split, dcfg["processed_dir"], dcfg["input_size"])
        img, label = ds[0]
        print(f"  {split:5s} | samples={len(ds):>7,} | img={tuple(img.shape)} | label={tuple(label.shape)}")
        assert img.shape == (3, dcfg["input_size"], dcfg["input_size"]), "Image shape mismatch"
        assert label.shape == (14,), "Label shape mismatch"
        assert img.dtype == torch.float32, "Image must be float32"
        assert label.dtype == torch.float32, "Label must be float32"

    # check normalization range (should be roughly -3 to 3)
    ds  = ChestMNISTDataset("train", dcfg["processed_dir"], dcfg["input_size"])
    imgs = torch.stack([ds[i][0] for i in range(32)])
    print(f"\n  Pixel stats (32 samples after normalize):")
    print(f"    mean={imgs.mean():.4f}  std={imgs.std():.4f}  min={imgs.min():.4f}  max={imgs.max():.4f}")
    print("  PASS ✓\n")


def check_model(cfg: dict, device: torch.device):
    print("── [2] Model Forward Pass ───────────────────────")
    dcfg = cfg["data"]
    model = build_model(cfg).to(device)
    model.eval()

    dummy = torch.randn(4, 3, dcfg["input_size"], dcfg["input_size"]).to(device)
    with torch.no_grad():
        out = model(dummy)

    print(f"  Input  : {tuple(dummy.shape)}")
    print(f"  Output : {tuple(out.shape)}")
    assert out.shape == (4, 14), f"Expected (4, 14), got {out.shape}"
    print("  PASS ✓\n")
    return model


def check_train_step(cfg: dict, model, device: torch.device):
    print("── [3] Train Step (2 batches) ───────────────────")
    import torch.nn as nn
    dcfg = cfg["data"]

    ds     = ChestMNISTDataset("train", dcfg["processed_dir"], dcfg["input_size"])
    loader = DataLoader(ds, batch_size=dcfg["batch_size"], shuffle=True,
                        num_workers=0)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    model.train()
    prev_loss = None
    for i, (imgs, labels) in enumerate(loader):
        if i == 2:
            break
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        print(f"  Batch {i+1} | loss={loss.item():.4f}")
        assert torch.isfinite(loss), "Loss is NaN or Inf!"
        prev_loss = loss.item()

    print("  PASS ✓\n")


def check_val_step(cfg: dict, model, device: torch.device):
    print("── [4] Val Step (2 batches) ─────────────────────")
    import torch.nn as nn
    dcfg = cfg["data"]

    ds     = ChestMNISTDataset("val", dcfg["processed_dir"], dcfg["input_size"])
    loader = DataLoader(ds, batch_size=dcfg["batch_size"] * 2, shuffle=False,
                        num_workers=0)
    criterion = nn.BCEWithLogitsLoss()

    model.eval()
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            if i == 2:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            probs  = torch.sigmoid(logits)
            print(f"  Batch {i+1} | loss={loss.item():.4f} | "
                  f"pred range=[{probs.min():.3f}, {probs.max():.3f}]")
            assert torch.isfinite(loss), "Val loss is NaN or Inf!"

    print("  PASS ✓\n")


def main(config_path: str):
    cfg    = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Experiment : {cfg['experiment']['name']}\n")

    check_data(cfg, device)
    model = check_model(cfg, device)
    check_train_step(cfg, model, device)
    check_val_step(cfg, model, device)

    print("══ ALL CHECKS PASSED ════════════════════════════")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    main(args.config)
