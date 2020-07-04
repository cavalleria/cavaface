## cavaface.pytorch: A Pytorch Training Framework for Deep Face Recognition
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
  * **Backone**
    * [x] ResNet(IR-SE)
    * [x] ResNeXt
    * [x] DenseNet
    * [x] MobileFaceNet
    * [x] MobileNetV3
    * [x] EfficientNet
    * [x] ProxylessNas
    * [x] GhostNet
    * [x] AttentionNet-IRSE
    * [x] EfficientPolyFace
    * [x] ResNeSt
    * [x] ReXNet
  * **Attention Module**
    * [x] SE
    * [x] CBAM
    * [x] ECA
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
    * [x] Optim(SGD,Adam,RAdam,[LookAhead](https://github.com/lonePatient/lookahead_pytorch),Ranger,AdamP,SGDP)
  * **[Data Augmentation](https://github.com/albumentations-team/albumentations)**
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
1. Install pytorch==1.4.0 and torchvision==0.5.0 following [official instruction](https://pytorch.org/).
2. Clone this repo, and we'll call the directory that you cloned as ${WORK_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Data preparation
**For training data**, please download the ms1m-retinaface in https://github.com/deepinsight/insightface/tree/master/iccv19-challenge.

**For test data**, please download the megaface and ijbc in https://github.com/deepinsight/insightface/tree/master/Evaluation.

### Training and Testing

#### Training on ms1m-retinaface
```
You can change the experimental setting by simply modifying the parameter in the config.py
bash train.sh
# To evaluate the model on validation set:
(1)please first download the val data in https://github.com/ZhaoJ9014/face.evoLVe.PyTorch.
(2)set the checkpoint dir in config.py
bash evaluate.sh
```
#### Testing on Megaface and IJBC
1. Put the test data and image list into proper directory.
2. Start evaluation service.
```
nohup python evaluate_service.py > logs/log.service &
```
3. Start extracting features and evaluating.
```
nohup bash run.sh > logs/log &
```

## Benchmark
| Backbone | Head | Loss | Flops | Megaface(Id/ver@1e-6) | IJBC(tar@far=1e-4) |
| :----: | :----: | :----: | :----: | :----: | :----: |
| MobileFaceNet | Arcface | Softmax | 450M | 92.8694/93.6329 | 92.80 |
| MobileFaceNet_ECA | Arcface | Softmax | 450M | 93.3329/94.4153 | 93.36 |
| MobileFaceNet_SE | Arcface | Softmax | 450M | 94.0951/94.4687 | 93.57 |
| MobileFaceNet_CBAM | Arcface | Softmax | 450M | 93.3068/94.3346 | 93.53 |
| GhostNet | Arcface | Softmax | 270M | 93.3914/94.3359 | 93.50 |
| [GhostNet_x1.3](https://drive.google.com/file/d/1KVgXIJo2Ym0Ffp3yK9FrIaiqjdAr2KFX/view?usp=sharing) | Arcface | Softmax | 440M | 95.3005/95.7757 | 94.27 |
| MobileNetV3 | Arcface | Softmax | 430M | 93.9805/95.7314 | 93.57 |
| ProxylessNAS_mobile | Arcface | Softmax | 630M | 93.2886/95.2094 | 93.74 |
| ProxylessNAS_cpu | Arcface | Softmax | 860M | 95.4242/95.79 | 94.22 |
| EfficientNet_b0 | Arcface | Softmax | 770M | 96.3589/97.1946 | 94.84 |
| EfficientNet_b1 | Arcface | Softmax | 1.14G | 97.095/97.4003 | 95.38 |
| AttentionNet-IRSE-92 | MV-AM | Softmax | 17.63G | 99.1356/99.3999 | 96.56 |
| IR-SE-100 | Arcface | Softmax | 24.18G | 99.0881/99.4259 | 96.69 |
| IR-SE-100 | ArcNegface | Softmax | 24.18G | 99.1304/98.7099 | 96.81 |
| IR-SE-100 | Curricularface | Softmax| 24.18G | 99.0497/98.6162 | 97.00 |
| [IR-SE-100](https://drive.google.com/file/d/1HdXgFmyMX4MGETTx6ACmx8AB-v79hrhp/view?usp=sharing) | Combined | Softmax| 24.18G | 99.0718/99.4493 | 96.83 |
| IR-SE-100 | CircleLoss | Softplus| 24.18G | 98.5732/98.4834 | 96.52 |
| ResNeSt-101 | Arcface | Softmax| 18.45G | 98.8746/98.5615 | 96.63 |
| DenseNet-201 | Arcface | Softmax| 8.52G | 98.3649/98.4294 | 96.03 |






## Acknowledgement

* This repo is modified and adapted on these great repositories [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch), [CurricularFace](https://github.com/HuangYG123/CurricularFace), [insightface](https://github.com/deepinsight/insightface) and [imgclsmob](https://github.com/osmr/imgclsmob/)
* The evaluation tools is developed by [Charrin](https://github.com/Charrin)

## Contact

```
cavallyb@gmail.com
```


