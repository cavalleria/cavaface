# cavaface.pytorch: A Pytorch Training Framework for Deep Face Recognition

[![python-url](https://img.shields.io/badge/Python-3.x-red.svg)](https://www.python.org/)
[![pytorch-url](https://img.shields.io/badge/Pytorch-1.4-blue.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HitCount](http://hits.dwyl.com/cavalleria/cavafacepytorch.svg)](http://hits.dwyl.com/cavalleria/cavafacepytorch)

By Yaobin Li and Liying Chi

## Introduction

This repo provides a high-performance distribute parallel training framework for face recognition with pytorch, including various backbones (e.g., ResNet, IR, IR-SE, ResNeXt, AttentionNet-IR-SE, ResNeSt, HRNet, etc.), various losses (e.g., Softmax, Focal, SphereFace, CosFace, AmSoftmax, ArcFace, ArcNegFace, CurricularFace, Li-Arcface, QAMFace, etc.), various data augmentation(e.g., RandomErasing, Mixup, RandAugment, Cutout, CutMix, etc.) and bags of tricks for improving performance (e.g., FP16 training(apex), Label smooth, LR warmup, etc)

## Main requirements

* **torch == 1.4.0**
* **torchvision == 0.5.0**
* **tensorboardX == 1.7**
* **bcolz == 1.2.1**
* **Python 3**
* **Apex == 0.1**

## Features

* **Backbone**
  * [x] ResNet(IR-SE)
  * [x] ResNeXt
  * [x] DenseNet
  * [x] MobileFaceNet
  * [x] MobileNetV3
  * [x] EfficientNet
  * [x] ProxylessNas
  * [x] GhostNet
  * [x] AttentionNet-IRSE
  * [x] ResNeSt
  * [x] ReXNet
  * [x] MobileNetV2
  * [x] MobileNeXt
* **Attention Module**
  * [x] SE
  * [x] CBAM
  * [x] ECA
  * [x] GCT
* **Loss**
  * [x] Softmax
  * [x] SphereFace
  * [x] Am_Softmax
  * [x] CosFace
  * [x] ArcFace
  * [x] Combined Loss
  * [x] AdaCos
  * [x] SV-X-Softmax
  * [x] CurricularFace
  * [x] ArcNegFace
  * [x] Li-Arcface
  * [x] QAMFace
  * [x] Circle Loss 
* **Parallel Training**
  * [x] Data Parallel
  * [x] Model Parallel
* **Automatic Mixed Precision**
  * [x] Apex
* **Optimizer**
  * [x] LR_Scheduler([faireq](https://github.com/pytorch/fairseq/tree/master/fairseq/optim/lr_scheduler),[rwightman](https://github.com/rwightman/pytorch-image-models/tree/master/timm/scheduler))
  * [x] Optim(SGD,Adam,RAdam,LookAhead,Ranger,AdamP,SGDP)
* **[Data Augmentation](https://github.com/albumentations-team/albumentations)**
  * [x] [RandomErasing](https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py)(官方版torchvision.transforms.RandomErasing)
  * [x] Mixup
  * [x] RandAugment
  * [x] Cutout
  * [x] CutMix
  * [x] Colorjitter 
* **Distillation**
  * [ ] KnowledgeDistillation
  * [ ] Multi Feature KD
* **Bag of Tricks**
  * [x] Label smooth
  * [x] LR warmup
  * [ ] Zero gamma

## Installation

See [INSTALL.md](https://github.com/cavalleria/cavaface.pytorch/blob/master/docs/INSTALL.md).

## Quick start

See [GETTING_STARTED.md](https://github.com/cavalleria/cavaface.pytorch/blob/master/docs/GETTING_STARTED.md).

## Model Zoo and Baselines

See [MODEL_ZOO.md](https://github.com/cavalleria/cavaface.pytorch/blob/master/docs/MODEL_ZOO.md).

## License

cavaface.pytorch is released under the [MIT license](https://github.com/cavalleria/cavaface.pytorch/blob/master/docs/LICENSE).

## Acknowledgement

* This repo is modified and adapted on these great repositories [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch), [CurricularFace](https://github.com/HuangYG123/CurricularFace), [insightface](https://github.com/deepinsight/insightface) and [imgclsmob](https://github.com/osmr/imgclsmob/)
* The evaluation tools is developed by [Charrin](https://github.com/Charrin)

## Contact

```markdown
cavallyb@gmail.com
```
