# Getting Start

### Data preparation

**For training data**, please download the ms1m-retinaface in https://github.com/deepinsight/insightface/tree/master/iccv19-challenge.

**For test data**, please download the megaface and ijbc in https://github.com/deepinsight/insightface/tree/master/Evaluation.

We provide our ms1m_retianface clean list. [Download](https://drive.google.com/file/d/165O76LkDBwo-PnCuTIdn45sXxWWhOBFh/view?usp=sharing)

### Training and Testing

#### Training on ms1m-retinaface

```bash
You can change the experimental setting by simply modifying the parameter in the config.py
bash train.sh
```

#### Testing on Megaface and IJBC

1. Put the test data and image list into proper directory.
2. Start evaluation service.

    ```python
    nohup python evaluate_service.py > logs/log.service &
    ```

3. Start extracting features and evaluating.

    ```bash
    nohup bash run.sh > logs/log &
    ```
