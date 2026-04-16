import timm
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "results" / "figures" / "eda_raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "mobilenetv4":    "mobilenetv4_conv_medium.e500_r224_in1k",
    "efficientnetv2": "efficientnetv2_s.in21k_ft_in1k",
    "ghostnetv3":     "ghostnetv3_100.in1k",
}

INPUT_SIZE = 224

lines = ["=== Model Data Config (via timm) ===\n"]

for name, timm_id in MODELS.items():
    model = timm.create_model(timm_id, pretrained=False)
    cfg   = timm.data.resolve_model_data_config(model)

    mean        = cfg["mean"]
    std         = cfg["std"]
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
