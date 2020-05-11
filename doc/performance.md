
## cavaface.pytorch: A Pytorch Training Framework for Deep Face Recognition

# Training speed

## MobileNet

| Apex |   Batchsize  | GPU  | GPU Memory | Samples/s |
| :----: | :----: | :----: | :----: | :----: |
| no   | 1024(2x512) | 2 x V100 | 2x26607MiB | 2840 |
| no   |  | 8 x V100 |    |  |
| no   | 1024(8x128) | 8 x Titan X | 8x6989MiB | 2250 |
| 00   | 1024(8x128) | 8 x Titan X |  |  |
| 01   | 1024(8x128) | 8 x Titan X |  |  |
| 02   | 1024(8x128) | 8 x Titan X |  |  |
| 03   | 1024(8x128) | 8 x Titan X |  |  |
| no   | 512(4x128) | 4 x Titan X | 4x6975MiB | 1700 |
| 00   | 512(4x128) | 4 x Titan X | 4x9537MiB | 1950 |
| 01   | 512(4x128) | 4 x Titan X | 4x4641MiB | 1757 |
| 02   | 512(4x128) | 4 x Titan X | 4x5567MiB | 1946 |
| 03   | 512(4x128) | 4 x Titan X | 4x4005MiB | 2138 |
| no   | 256(2x128) | 2 x Titan X | 2x7759MiB | 1223 |
| 00   | 256(2x128) | 2 x Titan X | 2x7577MiB | 1205 |
| 01   | 256(2x128) | 2 x Titan X | 2x4641MiB | 1190 |
| 02   | 256(2x128) | 2 x Titan X | 2x4195MiB | 1299 |
| 03   | 256(2x128) | 2 x Titan X | 2x4005MiB | 1413 |
