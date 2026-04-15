# Model Architecture Summary
Source: `report/model_architecture.txt`

---

## Summary Table

| Property | MobileNetV3 | EfficientNetV2 | ConvNeXtV2 |
|---|---|---|---|
| **timm ID** | `mobilenetv3_large_100` | `tf_efficientnetv2_b0` | `convnextv2_base.fcmae_ft_in22k_in1k` |
| **Parameters** | 5.5M | 7.1M | 88.7M |
| **Input Size** | 224×224 | 224×224 | 224×224 |
| **First Layer** | Conv2d(3→16, 3×3, s=2) | Conv2dSame(3→32, 3×3, s=2) | Conv2d(3→128, 4×4, s=4) |
| **Core Block** | InvertedResidual + DepthwiseSeparableConv | EdgeResidual → InvertedResidual | ConvNeXtBlock (DW 7×7) |
| **Normalization** | BatchNormAct2d | BatchNormAct2d | LayerNorm / LayerNorm2d |
| **Activation** | ReLU / Hardswish | SiLU | GELU |
| **Attention** | SqueezeExcite (Hardsigmoid) | SqueezeExcite (Sigmoid) | GlobalResponseNorm (GRN) |
| **Classifier Head** | GAP → Conv(960→1280) → Linear(1280→N) | Conv(192→1280) → GAP → Linear(1280→N) | GAP → LayerNorm → Linear(1024→N) |
| **Pretrain** | ImageNet-1k | ImageNet-1k (TF ported) | FCMAE → ImageNet-22k → ImageNet-1k |
| **Design Goal** | Mobile / edge efficiency | Balanced accuracy + efficiency | SOTA accuracy, Transformer-scale CNN |

---

## Per-Model Notes

### MobileNetV3 (`mobilenetv3_large_100`)
- Depthwise separable convolutions throughout — drastically cuts multiply-adds vs standard Conv2d
- SE blocks appear selectively in later stages (5×5 DW blocks), not every layer
- Dual activations: ReLU in early stages → Hardswish in later stages (hardware-friendly approximation of Swish)
- Lightest model — 5.5M params, fastest inference

### EfficientNetV2 (`tf_efficientnetv2_b0`)
- Early stages use **EdgeResidual** (no depthwise, fused Conv) → faster on accelerators than MBConv
- Later stages switch to standard **InvertedResidual** (MBConv) with SE
- SiLU (Swish) throughout — smooth gradient, slightly slower than ReLU but better accuracy
- Middle ground: 7.1M params, better accuracy than MobileNetV3

### ConvNeXtV2 (`convnextv2_base.fcmae_ft_in22k_in1k`)
- Stem is aggressive 4×4 stride-4 patch embed — Transformer-style tokenization, not gradual downsampling
- Depthwise 7×7 Conv in every block — large receptive field per layer
- **GRN (Global Response Normalization)** in MLP: normalizes across channels globally, prevents feature collapse during MAE pretraining
- LayerNorm instead of BatchNorm — more stable across varied batch sizes
- Heaviest: 88.7M params, highest accuracy ceiling
- Pretrained with FCMAE (masked autoencoder) + ImageNet-22k fine-tune → strongest transfer starting point

---

## Complexity vs Accuracy Trade-off

```
Accuracy potential ──────────────────────────────►
MobileNetV3 (5.5M) ──► EfficientNetV2 (7.1M) ──► ConvNeXtV2 (88.7M)
◄────────────────────────────────────────────── Speed / Inference time
```
