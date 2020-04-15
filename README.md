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

* This repo is inspired by [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch), [CurricularFace](https://github.com/HuangYG123/CurricularFace), [insightface](https://github.com/deepinsight/insightface).


## Contact

```
cavalleria@gmail.com
```


