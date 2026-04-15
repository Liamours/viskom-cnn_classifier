import timm
import torch

models = {
    "mobilenetv3": "mobilenetv3_large_100",
    "convnextv2": "convnextv2_base.fcmae_ft_in22k_in1k",
    "efficientnetv2": "tf_efficientnetv2_b0"
}

for label, model_name in models.items():
    print(f"{'='*20} {label.upper()} ({model_name}) {'='*20}")
    model = timm.create_model(model_name, pretrained=False)
    print(model)
    print("\n")