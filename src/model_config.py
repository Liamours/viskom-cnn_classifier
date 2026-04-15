import timm
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "results" / "figures" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "mobilenetv3":  "mobilenetv3_large_100",
    "efficientnetv2": "tf_efficientnetv2_b0",
    "convnextv2":   "convnextv2_base.fcmae_ft_in22k_in1k",
}

INPUT_SIZE = 224

lines = ["=== Model Data Config (via timm) ===\n"]

for name, timm_id in MODELS.items():
    model = timm.create_model(timm_id, pretrained=False)
    cfg   = timm.data.resolve_model_data_config(model)

    mean       = cfg["mean"]
    std        = cfg["std"]
    native_size = cfg["input_size"]

    print(f"\n{name} ({timm_id})")
    print(f"  mean        : {mean}")
    print(f"  std         : {std}")
    print(f"  native size : {native_size}")
    print(f"  used size   : (3, {INPUT_SIZE}, {INPUT_SIZE})")

    lines += [
        f"Model       : {name}",
        f"timm ID     : {timm_id}",
        f"mean        : {mean}",
        f"std         : {std}",
        f"native size : {native_size}",
        f"used size   : (3, {INPUT_SIZE}, {INPUT_SIZE})",
        "",
    ]

out_path = OUT_DIR / "model_data_config.txt"
out_path.write_text("\n".join(lines))
print(f"\nSaved: {out_path}")
