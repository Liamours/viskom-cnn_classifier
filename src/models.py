import torch
import torch.nn as nn
import timm
from pathlib import Path


def build_model(cfg: dict) -> nn.Module:
    mcfg = cfg["model"]
    fcfg = cfg["finetune"]

    model = timm.create_model(
        mcfg["timm_id"],
        pretrained=False,
        num_classes=mcfg["num_classes"],
    )

    weights_path = mcfg.get("pretrained_weights")
    if weights_path and Path(weights_path).exists():
        state = torch.load(weights_path, map_location="cpu")
        # strip classifier head from saved weights (size mismatch on num_classes)
        head_keys = [k for k in state if k.startswith(("classifier", "head", "fc"))]
        for k in head_keys:
            state.pop(k, None)
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights: {weights_path}")
    else:
        print("No local weights found — using random init (run download_weights.py first)")

    _apply_freeze(model, fcfg)
    return model


def _apply_freeze(model: nn.Module, fcfg: dict):
    if not fcfg["enabled"]:
        _freeze_backbone(model)
        print("Finetune disabled — backbone frozen, head trainable only")
        return

    ratio = float(fcfg["unfreeze_ratio"])
    ratio = max(0.0, min(1.0, ratio))

    all_params = list(model.named_parameters())
    n_total    = len(all_params)
    n_unfreeze = int(n_total * ratio)

    # freeze all first
    for _, p in all_params:
        p.requires_grad = False

    # unfreeze last N params (closest to output = fine-tuning standard)
    for _, p in all_params[n_total - n_unfreeze:]:
        p.requires_grad = True

    # always keep head trainable
    _unfreeze_head(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Finetune: {ratio:.0%} backbone unfrozen — "
          f"{trainable:,} / {total:,} params trainable")


def _freeze_backbone(model: nn.Module):
    for name, p in model.named_parameters():
        is_head = any(name.startswith(k) for k in ("classifier", "head", "fc"))
        p.requires_grad = is_head


def _unfreeze_head(model: nn.Module):
    for name, p in model.named_parameters():
        if any(name.startswith(k) for k in ("classifier", "head", "fc")):
            p.requires_grad = True
