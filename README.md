## cavaface.pytorch: A Pytorch Training Framework for Deep Face Recognition

By Yaobin Li


## License

The code of cavaface.pytorch is released under the MIT License. There is no limitation for both acadmic and commercial usage.

The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only.

## Main requirements

  * **torch == 1.1.0**
  * **torchvision == 0.3.0**
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
    * [ ] AttentionNet-IRSE
    * [x] EfficientPolyFace
    * [ ] Res2Net
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
    * [ ] Optim(SGD,Adam)
  * **[Data Augmentation](https://github.com/albumentations-team/albumentations)**
    * [ ] Blur
    * [ ] Motion
    * [ ] Occlusion
    * [ ] Color jitter
    * [ ] [RandomErasing](https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py)(官方版torchvision.transforms.RandomErasing)
    * [ ] [AutoAugment](https://github.com/rwightman/pytorch-image-models)
    * [ ] Mixup
    * [ ] RandAugment
    * [ ] AugMix
  * **Distillation**
    * [ ] KnowledgeDistillation
    * [ ] Multi Feature KD
  * **Bag of Tricks**
    * [ ] Label smooth
    * [ ] LR warmup
    * [ ] Zero gamma

## Usage
```bash
# To train the model:
sh train.sh
# To evaluate the model:
(1)please first download the val data in https://github.com/ZhaoJ9014/face.evoLVe.PyTorch.
(2)set the checkpoint dir in config.py
sh evaluate.sh
```
You can change the experimental setting by simply modifying the parameter in the config.py


## Acknowledgement

* This repo is modified and adapted on these great repositories [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch), [CurricularFace](https://github.com/HuangYG123/CurricularFace), [insightface](https://github.com/deepinsight/insightface) and [imgclsmob](https://github.com/osmr/imgclsmob/)


## Contact

```
cavalleria@gmail.com
```


