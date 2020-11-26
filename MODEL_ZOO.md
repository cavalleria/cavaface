# Model Zoo and Baselines

## Benchmark

| Backbone | Head | Loss | Flops/Params | Megaface(Id/ver@1e-6) | IJBC(tar@far=1e-4) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| MobileFaceNet | Arcface | Softmax | 450M/1.20M | 92.8694/93.6329 | 92.80 |
| MobileNetV2 | Arcface | Softmax | 430M/2.26M | 92.8128/93.7644 | 93.30 |
| MobileNeXt | Arcface | Softmax | 420M/1.86M | 93.3368/94.6913 | 93.38 |
| MobileFaceNet_ECA | Arcface | Softmax | 450M/1.20M | 93.7624/95.2328 | 93.40 |
| MobileFaceNet_SE | Arcface | Softmax | 450M/1.23M | 94.0951/94.4687 | 93.57 |
| MobileFaceNet_CBAM | Arcface | Softmax | 450M/1.23M | 93.3068/94.3346 | 93.53 |
| MobileFaceNet_GCT | Arcface | Softmax | 450M/1.21M | 93.133/94.1836 | 93.09 |
| GhostNet | Arcface | Softmax | 270M/2.49M | 93.3914/94.3359 | 93.50 |
| [GhostNet_x1.3](https://drive.google.com/file/d/1KVgXIJo2Ym0Ffp3yK9FrIaiqjdAr2KFX/view?usp=sharing) | Arcface | Softmax | 440M/4.06M | 95.3005/95.7757 | 94.27 |
| MobileNetV3 | Arcface | Softmax | 430M/3.2M | 93.9805/95.7314 | 93.57 |
| ProxylessNAS_mobile | Arcface | Softmax | 630M/2.84M | 93.2886/95.2094 | 93.74 |
| ProxylessNAS_cpu | Arcface | Softmax | 860M/2.89M | 95.4242/95.79 | 94.22 |
| EfficientNet_b0 | Arcface | Softmax | 770M/4.07M | 96.3589/97.1946 | 94.84 |
| EfficientNet_b1 | Arcface | Softmax | 1.14G/6.58M | 97.095/97.4003 | 95.38 |
| AttentionNet-IRSE-92 | MV-AM | Softmax | 17.63G/55.42M | 99.1356/99.3999 | 96.56 |
| IR-SE-100 | Arcface | Softmax | 24.18G/65.5M | 99.0881/99.4259 | 96.69 |
| IR-SE-100 | ArcNegface | Softmax | 24.18G/65.5M  | 99.1304/98.7099 | 96.81 |
| IR-SE-100 | Curricularface | Softmax| 24.18G/65.5M  | 99.0497/98.6162 | 97.00 |
| [IR-SE-100](https://drive.google.com/file/d/1HdXgFmyMX4MGETTx6ACmx8AB-v79hrhp/view?usp=sharing) | Combined | Softmax| 24.18G/65.5M | 99.0718/99.4493 | 96.83 |
| IR-SE-100 | CircleLoss | Softplus| 24.18G/65.5M  | 98.5732/98.4834 | 96.52 |
| ResNeSt-101 | Arcface | Softmax| 18.45G/97.61M | 98.8746/98.5615 | 96.63 |
| DenseNet-201 | Arcface | Softmax| 8.52G/66.37M | 98.3649/98.4294 | 96.03 |

## Data augmentation

| Backbone | Head | DataAugment | Megaface(Id/ver@1e-6) | IJBC(tar@far=1e-4) |
| :----: | :----: | :----: | :----: | :----: |
| MobileFaceNet | Arcface | Basline | 92.8694/93.6329 | 92.80 |
| MobileFaceNet | Arcface | Randaug | 88.9829/91.045 | 80.86 |
| MobileFaceNet | Arcface | Mixup | 90.8965/92.6123 | 91.18 |
| MobileFaceNet | Arcface | RandErasing | 91.2396/93.0953 | 92.06 |
| MobileFaceNet | Arcface | Cutout | 90.7761/92.1215 | 81.02 |
| MobileFaceNet | Arcface | Cutmix | 88.9497/90.3823 | 89.64 |
| MobileFaceNet | Arcface | ColorJitter | 91.9237/92.8649 | 92.03 |

## Training Speed Benchmark

Device: 8 * Tesla V100-PCIE-32GB
| Backbone | Dataset | Batch Size | FP16 | Samples/s |
| :----: | :----: | :----: | :----: | :----: |
| MobileFaceNet | MS1MV3 | 8x128 | False | 7065 |
| MobileFaceNet | MS1MV3 | 8x128 | True | 10240 |
| IR_SE_100 | MS1MV3 | 8x128 | False | 1454 |
| IR_SE_100 | MS1MV3 | 8x128 | True | 2662 |
| IR_100 | MS1MV3 | 8x128 | False | 1618 |
| IR_100 | MS1MV3 | 8x128 | True | 3173 |
| IR_100 | Glint360 | 8x128 | False | 1413 |
| IR_100 | Glint360 | 8x128 | True | 2682 |




