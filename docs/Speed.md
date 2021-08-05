# Training Speed Benchmark


## Training speed 

Tesla V100-PCIE-32GB

| Backbone | Dataset | Batch Size | FP16 | Samples/s |
| :----: | :----: | :----: | :----: | :----: |
| MobileFaceNet | MS1MV3 | 8x128 | False | 7065 |
| MobileFaceNet | MS1MV3 | 8x128 | True | 10240 |
| IR_SE_100 | MS1MV3 | 8x128 | False | 1454 |
| IR_SE_100 | MS1MV3 | 8x128 | True | 2662 |
| IR_100 | MS1MV3 | 8x128 | False | 1618 |
| IR_100 | MS1MV3 | 8x128 | True | 3173 |
| IR_100 | Glint360k | 8x128 | False | 1413 |
| IR_100 | Glint360k | 8x128 | True | 2682 |

Tesla V100 SMX2 32G

NVIDIA-SMI 465.19.01    Driver Version: 465.19.01    CUDA Version: 11.3

ngc_pytorch_21.02py3:latest

| classes |	backbone | batchsize | amp | speed | gpu memory |
| :----: | :----: | :----: | :----: | :----: | :----: |
| 10w	| R50	| 1024	| no	| 3010	| 15742 |
|	    | R50	| 1024	| O1	| 5427	| 10458 |
| 30w	| R50	| 1024	| no	| 2723	| 18172 |
|	    | R50	| 1024	| O1	| 4638	| 13660 |
| 50w	| R50	| 1024	| no	| 2488	| 22894 |
|       | R50	| 1024	| O1	| 4014	| 17682 |
| 100w	| R50	| 1024	| no	| -	| - |
|	    | R50	| 1024	| O1	| 2969	| 27444 |
|	    | R50	| 1024	| O2	| 3471	| 21108 |
| 130w	| R50	| 1024	| O1	| 2570	| 31410 |
|	    | R50	| 1024	| O2	| 3051	| 29344 |
| 160w	| R50	| 1024	| O1	| -	| - |
|    	| R50	| 1024	| O2	| 2723	| 32032 |
| 100w	| R100	| 1024	| O1	| 2252	| 31798 |
|	    | R100	| 1024	| O2	| 2529	| 25388 |
| 130w	| R100	| 1024	| O1	| -	| - |
|	    | R100	| 1024	| O2	| 2304	| 31932 |