## cavaface.pytorch: A Pytorch Training Framework for Deep Face Recognition

By Yaobin Li

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

## Usage

### For training data
please download the ms1m-retinaface in https://github.com/deepinsight/insightface/tree/master/iccv19-challenge.
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






## Acknowledgement

* This repo is modified and adapted on these great repositories [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch), [CurricularFace](https://github.com/HuangYG123/CurricularFace), [insightface](https://github.com/deepinsight/insightface) and [imgclsmob](https://github.com/osmr/imgclsmob/)


## Contact

```
cavallyb@gmail.com
```


