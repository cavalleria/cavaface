## cavaface.pytorch: A Pytorch Training Framework for Deep Face Recognition
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HitCount](http://hits.dwyl.com/cavalleria/cavafacepytorch.svg)](http://hits.dwyl.com/cavalleria/cavafacepytorch)

By Yaobin Li

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
  * **Backone**
    * [x] ResNet(IR-SE)
    * [x] ResNeXt
    * [ ] DenseNet
    * [x] MobileFaceNet
    * [ ] MobileNetV3
    * [ ] EfficientNet
    * [ ] VargFaceNet
    * [ ] ProxylessNas
    * [ ] GhostNet
    * [x] AttentionNet-IRSE
    * [x] EfficientPolyFace
    * [x] ResNeSt
  * **Attention Module**
    * [x] SE
    * [ ] CBAM
    * [ ] ECA
    * [ ] ACNet
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
  * **Parallel Training**
    * [x] Data Parallel
    * [x] Model Parallel
  * **Automatic Mixed Precision**
    * [x] Apex
  * **Optimizer**
    * [x] LR_Scheduler([faireq](https://github.com/pytorch/fairseq/tree/master/fairseq/optim/lr_scheduler),[rwightman](https://github.com/rwightman/pytorch-image-models/tree/master/timm/scheduler))
    * [ ] Optim(SGD,Adam,[LookAhead](https://github.com/lonePatient/lookahead_pytorch))
  * **[Data Augmentation](https://github.com/albumentations-team/albumentations)**
    * [ ] Blur
    * [x] [RandomErasing](https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py)(官方版torchvision.transforms.RandomErasing)
    * [x] Mixup
    * [x] RandAugment
    * [x] Cutout
    * [x] CutMix
  * **Distillation**
    * [ ] KnowledgeDistillation
    * [ ] Multi Feature KD
  * **Bag of Tricks**
    * [x] Label smooth
    * [x] LR warmup
    * [ ] Zero gamma

## Quick start
### Installation
1. Install pytorch==1.4.0 following and torchvision==0.5.0[official instruction](https://pytorch.org/).
2. Clone this repo, and we'll call the directory that you cloned as ${WORK_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Data preparation
please download the ms1m-retinaface in https://github.com/deepinsight/insightface/tree/master/iccv19-challenge.

### Training and Testing
```bash
# To train the model:
sh train.sh
# To evaluate the model:
(1)please first download the val data in https://github.com/ZhaoJ9014/face.evoLVe.PyTorch.
(2)set the checkpoint dir in config.py
sh evaluate.sh
```
You can change the experimental setting by simply modifying the parameter in the config.py

## Benchmark
| Backbone | Head | Loss | Flops | Megaface(Id/ver@1e-6) | IJBC(tar@far=1e-4) |
| :----: | :----: | :----:| :----: | :----: | :----: |
| MobileFaceNet | Arcface | Softmax | 440M |  92.8694/93.6329 | 92.80 |
| AttentionNet-IRSE-92 | MV-AM | Softmax | 17.63G | 99.1356/99.3999 | 96.56 |
| IR-SE-100 | Arcface | Softmax | 24.18G | 99.0881/99.4259 | 96.69 |
| IR-SE-100 | ArcNegface | Softmax | 24.18G | 99.1304/98.7099 | 96.81 |
| IR-SE-100 | Curricularface | Softmax| 24.18G | 99.0497/98.6162 | 97.00 |
| IR-SE-100 | Combined | Softmax| 24.18G | 99.0718/99.4493 | 96.83 |
| ResNeSt-101 | Arcface | Softmax| 18.45G | 98.6279/98.7307 | 96.65 |






## Acknowledgement

* This repo is modified and adapted on these great repositories [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch), [CurricularFace](https://github.com/HuangYG123/CurricularFace), [insightface](https://github.com/deepinsight/insightface) and [imgclsmob](https://github.com/osmr/imgclsmob/)


## Contact

```
cavallyb@gmail.com
```


