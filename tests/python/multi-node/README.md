# Test multi-devices and multi-machines

Note that `CUDNN` leads to randomness, need to disable if comparing to the baseline

- `local_*` for multi-devices and single machine. Requires two GPUs.

- `dist_sync_*` for multi-machines with BSP synchronizations

`dist_async_*` for multi-machines with asynchronous SGD

```
ln -s ../../../ps-lite/tracker/dmlc_local.py .
./dmlc_local.py -n 2 -s 2 ./dist_sync_mlp.py
```

# Results

## cifar10, inceptions

single gtx 980. batch size = 128 and learning rate = .1

```
[03:42:04] src/io/iter_image_recordio.cc:212: ImageRecordIOParser: data/cifar/train.rec, use 4 threads for decoding..
[03:42:04] src/io/./iter_normalize.h:98: Load mean image from data/cifar/cifar_mean.bin
[03:42:04] src/io/iter_image_recordio.cc:212: ImageRecordIOParser: data/cifar/test.rec, use 4 threads for decoding..
[03:42:04] src/io/./iter_normalize.h:98: Load mean image from data/cifar/cifar_mean.bin
INFO:root:Iteration[0] Train-accuracy=0.523938
INFO:root:Iteration[0] Time cost=104.396
INFO:root:Iteration[0] Validation-accuracy=0.665941
INFO:root:Iteration[1] Train-accuracy=0.721108
INFO:root:Iteration[1] Time cost=105.245
INFO:root:Iteration[1] Validation-accuracy=0.755934
INFO:root:Iteration[2] Train-accuracy=0.793298
INFO:root:Iteration[2] Time cost=105.101
INFO:root:Iteration[2] Validation-accuracy=0.784909
INFO:root:Iteration[3] Train-accuracy=0.835198
INFO:root:Iteration[3] Time cost=104.816
INFO:root:Iteration[3] Validation-accuracy=0.799150
INFO:root:Iteration[4] Train-accuracy=0.869625
INFO:root:Iteration[4] Time cost=104.571
INFO:root:Iteration[4] Validation-accuracy=0.809533
INFO:root:Iteration[5] Train-accuracy=0.895201
INFO:root:Iteration[5] Time cost=104.357
INFO:root:Iteration[5] Validation-accuracy=0.811214
INFO:root:Iteration[6] Train-accuracy=0.911025
INFO:root:Iteration[6] Time cost=104.347
INFO:root:Iteration[6] Validation-accuracy=0.799644
INFO:root:Iteration[7] Train-accuracy=0.923853
INFO:root:Iteration[7] Time cost=104.108
INFO:root:Iteration[7] Validation-accuracy=0.806468
INFO:root:Iteration[8] Train-accuracy=0.936301
INFO:root:Iteration[8] Time cost=104.178
INFO:root:Iteration[8] Validation-accuracy=0.813687
INFO:root:Iteration[9] Train-accuracy=0.950068
INFO:root:Iteration[9] Time cost=104.522
INFO:root:Iteration[9] Validation-accuracy=0.820115
INFO:root:Accuracy = 0.820100
```

using 3x dual gtx 980 machines, async inception with batch size = 128 and
learning rate = .05


```
[03:23:29] src/io/iter_image_recordio.cc:212: ImageRecordIOParser: data/cifar/train.rec, use 4 threads for decoding..
[03:23:31] src/io/iter_image_recordio.cc:212: ImageRecordIOParser: data/cifar/train.rec, use 4 threads for decoding..
[03:23:29] src/io/./iter_normalize.h:98: Load mean image from data/cifar/cifar_mean.bin
[03:23:31] src/io/./iter_normalize.h:98: Load mean image from data/cifar/cifar_mean.bin
[03:23:30] src/io/iter_image_recordio.cc:212: ImageRecordIOParser: data/cifar/train.rec, use 4 threads for decoding..
[03:23:30] src/io/./iter_normalize.h:98: Load mean image from data/cifar/cifar_mean.bin
[03:23:29] src/io/iter_image_recordio.cc:212: ImageRecordIOParser: data/cifar/test.rec, use 4 threads for decoding..
[03:23:31] src/io/iter_image_recordio.cc:212: ImageRecordIOParser: data/cifar/test.rec, use 4 threads for decoding..
[03:23:29] src/io/./iter_normalize.h:98: Load mean image from data/cifar/cifar_mean.bin
[03:23:31] src/io/./iter_normalize.h:98: Load mean image from data/cifar/cifar_mean.bin
[03:23:30] src/io/iter_image_recordio.cc:212: ImageRecordIOParser: data/cifar/test.rec, use 4 threads for decoding..
[03:23:30] src/io/./iter_normalize.h:98: Load mean image from data/cifar/cifar_mean.bin
INFO:root:Start training with [gpu(0), gpu(1)]
INFO:root:Start training with [gpu(0), gpu(1)]
INFO:root:Start training with [gpu(0), gpu(1)]
INFO:root:Iteration[0] Train-accuracy=0.185276
INFO:root:Iteration[0] Time cost=21.556
INFO:root:Iteration[0] Train-accuracy=0.184255
INFO:root:Iteration[0] Time cost=22.021
INFO:root:Iteration[0] Train-accuracy=0.183834
INFO:root:Iteration[0] Time cost=22.342
INFO:root:Iteration[0] Validation-accuracy=0.225079
INFO:root:Iteration[0] Validation-accuracy=0.236452
INFO:root:Iteration[0] Validation-accuracy=0.237836
INFO:root:Iteration[1] Train-accuracy=0.308624
INFO:root:Iteration[1] Time cost=21.617
INFO:root:Iteration[1] Train-accuracy=0.312977
INFO:root:Iteration[1] Time cost=21.603
INFO:root:Iteration[1] Train-accuracy=0.309637
INFO:root:Iteration[1] Time cost=21.917
INFO:root:Iteration[1] Validation-accuracy=0.333169
INFO:root:Iteration[1] Validation-accuracy=0.382812
INFO:root:Iteration[1] Validation-accuracy=0.385186
INFO:root:Iteration[2] Train-accuracy=0.426885
INFO:root:Iteration[2] Time cost=21.531
INFO:root:Iteration[2] Train-accuracy=0.420802
INFO:root:Iteration[2] Time cost=21.469
INFO:root:Iteration[2] Train-accuracy=0.436844
INFO:root:Iteration[2] Time cost=22.053
INFO:root:Iteration[2] Validation-accuracy=0.487935
INFO:root:Iteration[2] Validation-accuracy=0.491495
INFO:root:Iteration[2] Validation-accuracy=0.532832
INFO:root:Iteration[3] Train-accuracy=0.541209
INFO:root:Iteration[3] Time cost=21.817
INFO:root:Iteration[3] Train-accuracy=0.544072
INFO:root:Iteration[3] Time cost=21.759
INFO:root:Iteration[3] Train-accuracy=0.546458
INFO:root:Iteration[3] Time cost=22.156
INFO:root:Iteration[3] Validation-accuracy=0.589102
INFO:root:Iteration[3] Validation-accuracy=0.559138
INFO:root:Iteration[3] Validation-accuracy=0.613528
INFO:root:Iteration[4] Train-accuracy=0.618500
INFO:root:Iteration[4] Time cost=21.552
INFO:root:Iteration[4] Train-accuracy=0.614862
INFO:root:Iteration[4] Time cost=21.544
INFO:root:Iteration[4] Train-accuracy=0.619573
INFO:root:Iteration[4] Time cost=21.890
INFO:root:Iteration[4] Validation-accuracy=0.630241
INFO:root:Iteration[4] Validation-accuracy=0.618176
INFO:root:Iteration[4] Validation-accuracy=0.666930
INFO:root:Iteration[5] Train-accuracy=0.673843
INFO:root:Iteration[5] Time cost=21.056
INFO:root:Iteration[5] Train-accuracy=0.675692
INFO:root:Iteration[5] Time cost=21.120
INFO:root:Iteration[5] Train-accuracy=0.678912
INFO:root:Iteration[5] Time cost=21.721
INFO:root:Iteration[5] Validation-accuracy=0.657634
INFO:root:Iteration[5] Validation-accuracy=0.677809
INFO:root:Iteration[5] Validation-accuracy=0.715882
INFO:root:Iteration[6] Train-accuracy=0.722149
INFO:root:Iteration[6] Time cost=20.579
INFO:root:Iteration[6] Train-accuracy=0.724833
INFO:root:Iteration[6] Time cost=20.548
INFO:root:Iteration[6] Train-accuracy=0.720241
INFO:root:Iteration[6] Time cost=20.772
INFO:root:Iteration[6] Validation-accuracy=0.692939
INFO:root:Iteration[6] Validation-accuracy=0.714794
INFO:root:Iteration[6] Validation-accuracy=0.748220
INFO:root:Iteration[7] Train-accuracy=0.760854
INFO:root:Iteration[7] Time cost=20.801
INFO:root:Iteration[7] Train-accuracy=0.757276
INFO:root:Iteration[7] Time cost=21.080
INFO:root:Iteration[7] Validation-accuracy=0.735858
INFO:root:Iteration[7] Train-accuracy=0.758767
INFO:root:Iteration[7] Time cost=21.353
INFO:root:Iteration[7] Validation-accuracy=0.737638
INFO:root:Iteration[7] Validation-accuracy=0.774328
INFO:root:Iteration[8] Train-accuracy=0.794967
INFO:root:Iteration[8] Time cost=21.593
INFO:root:Iteration[8] Train-accuracy=0.798485
INFO:root:Iteration[8] Time cost=21.672
INFO:root:Iteration[8] Validation-accuracy=0.762460
INFO:root:Iteration[8] Train-accuracy=0.795503
INFO:root:Iteration[8] Time cost=22.155
INFO:root:Iteration[8] Validation-accuracy=0.745748
INFO:root:Iteration[8] Validation-accuracy=0.784513
INFO:root:Iteration[9] Train-accuracy=0.825561
INFO:root:Iteration[9] Time cost=21.644
INFO:root:Iteration[9] Train-accuracy=0.821923
INFO:root:Iteration[9] Time cost=21.479
INFO:root:Iteration[9] Validation-accuracy=0.727453
INFO:root:Iteration[9] Validation-accuracy=0.745253
INFO:root:Iteration[9] Train-accuracy=0.819716
INFO:root:Iteration[9] Time cost=21.927
INFO:root:Iteration[9] Validation-accuracy=0.781151
INFO:root:Iteration[10] Train-accuracy=0.842975
INFO:root:Iteration[10] Time cost=21.431
INFO:root:Iteration[10] Train-accuracy=0.841543
INFO:root:Iteration[10] Time cost=21.387
INFO:root:Iteration[10] Validation-accuracy=0.768196
INFO:root:Iteration[10] Validation-accuracy=0.781448
INFO:root:Iteration[10] Train-accuracy=0.843989
INFO:root:Iteration[10] Time cost=21.875
INFO:root:Iteration[10] Validation-accuracy=0.804391
INFO:root:Iteration[11] Train-accuracy=0.860329
INFO:root:Iteration[11] Time cost=20.664
INFO:root:Iteration[11] Train-accuracy=0.858958
INFO:root:Iteration[11] Time cost=20.734
INFO:root:Iteration[11] Validation-accuracy=0.780063
INFO:root:Iteration[11] Validation-accuracy=0.774426
INFO:root:Iteration[11] Train-accuracy=0.861104
INFO:root:Iteration[11] Time cost=21.449
INFO:root:Iteration[11] Validation-accuracy=0.818335
INFO:root:Iteration[12] Train-accuracy=0.885973
INFO:root:Iteration[12] Time cost=21.037
INFO:root:Iteration[12] Train-accuracy=0.887583
INFO:root:Iteration[12] Time cost=21.066
INFO:root:Iteration[12] Validation-accuracy=0.798358
INFO:root:Iteration[12] Validation-accuracy=0.803204
INFO:root:Iteration[12] Train-accuracy=0.885914
INFO:root:Iteration[12] Time cost=21.738
INFO:root:Iteration[12] Validation-accuracy=0.812203
INFO:root:Iteration[13] Train-accuracy=0.904103
INFO:root:Iteration[13] Time cost=21.326
INFO:root:Iteration[13] Train-accuracy=0.904282
INFO:root:Iteration[13] Time cost=21.278
INFO:root:Iteration[13] Validation-accuracy=0.791238
INFO:root:Iteration[13] Validation-accuracy=0.799842
INFO:root:Iteration[13] Train-accuracy=0.901002
INFO:root:Iteration[13] Time cost=21.408
INFO:root:Iteration[13] Validation-accuracy=0.802116
INFO:root:Iteration[14] Train-accuracy=0.911140
INFO:root:Iteration[14] Time cost=21.527
INFO:root:Iteration[14] Train-accuracy=0.913705
INFO:root:Iteration[14] Time cost=21.569
INFO:root:Iteration[14] Validation-accuracy=0.803204
INFO:root:Iteration[14] Validation-accuracy=0.803303
INFO:root:Iteration[14] Train-accuracy=0.914182
INFO:root:Iteration[14] Time cost=22.170
INFO:root:Iteration[14] Validation-accuracy=0.771460
INFO:root:Iteration[15] Train-accuracy=0.915852
INFO:root:Iteration[15] Time cost=21.608
INFO:root:Iteration[15] Train-accuracy=0.911975
INFO:root:Iteration[15] Time cost=21.623
INFO:root:Iteration[15] Validation-accuracy=0.801325
INFO:root:Iteration[15] Validation-accuracy=0.798259
INFO:root:Iteration[15] Train-accuracy=0.923008
INFO:root:Iteration[15] Time cost=21.806
INFO:root:Iteration[15] Validation-accuracy=0.809335
INFO:root:Iteration[16] Train-accuracy=0.938096
INFO:root:Iteration[16] Time cost=21.857
INFO:root:Iteration[16] Train-accuracy=0.944358
INFO:root:Iteration[16] Time cost=21.954
INFO:root:Iteration[16] Validation-accuracy=0.790249
INFO:root:Iteration[16] Validation-accuracy=0.795095
INFO:root:Iteration[16] Train-accuracy=0.947877
INFO:root:Iteration[16] Time cost=21.844
INFO:root:Iteration[16] Validation-accuracy=0.812797
INFO:root:Iteration[17] Train-accuracy=0.953006
INFO:root:Iteration[17] Time cost=21.357
INFO:root:Iteration[17] Train-accuracy=0.957121
INFO:root:Iteration[17] Time cost=21.431
INFO:root:Iteration[17] Validation-accuracy=0.793908
INFO:root:Iteration[17] Validation-accuracy=0.793216
INFO:root:Iteration[17] Train-accuracy=0.962846
INFO:root:Iteration[17] Time cost=21.819
INFO:root:Iteration[17] Validation-accuracy=0.812994
INFO:root:Iteration[18] Train-accuracy=0.961772
INFO:root:Iteration[18] Time cost=20.599
INFO:root:Iteration[18] Train-accuracy=0.963800
INFO:root:Iteration[18] Time cost=20.569
INFO:root:Iteration[18] Validation-accuracy=0.815467
INFO:root:Iteration[18] Validation-accuracy=0.818829
INFO:root:Iteration[18] Train-accuracy=0.966603
INFO:root:Iteration[18] Time cost=21.018
INFO:root:Iteration[18] Validation-accuracy=0.812698
INFO:root:Iteration[19] Train-accuracy=0.975131
INFO:root:Iteration[19] Time cost=20.671
INFO:root:Iteration[19] Train-accuracy=0.975847
INFO:root:Iteration[19] Time cost=20.758
INFO:root:Iteration[19] Validation-accuracy=0.822785
INFO:root:Iteration[19] Validation-accuracy=0.823378
INFO:root:Iteration[19] Train-accuracy=0.981990
INFO:root:Iteration[19] Time cost=20.912
INFO:root:Accuracy = 0.823800
INFO:root:Iteration[19] Validation-accuracy=0.828521
INFO:root:Accuracy = 0.829200
INFO:root:Accuracy = 0.833000
```

## imagenet

3 x dual 980, with cudnn, 1G ethernet

`dist_sync`:

```
INFO:root:Start training with [gpu(0), gpu(1)]
INFO:root:Start training with [gpu(0), gpu(1)]
INFO:root:Start training with [gpu(0), gpu(1)]
INFO:root:Iter[0] Batch [5]	Speed: 175.98 samples/sec
INFO:root:Iter[0] Batch [5]	Speed: 173.52 samples/sec
INFO:root:Iter[0] Batch [5]	Speed: 171.04 samples/sec
INFO:root:Iter[0] Batch [10]	Speed: 107.82 samples/sec
INFO:root:Iter[0] Batch [10]	Speed: 108.03 samples/sec
INFO:root:Iter[0] Batch [10]	Speed: 107.79 samples/sec
INFO:root:Iter[0] Batch [15]	Speed: 109.53 samples/sec
INFO:root:Iter[0] Batch [15]	Speed: 109.74 samples/sec
INFO:root:Iter[0] Batch [15]	Speed: 110.21 samples/sec
INFO:root:Iter[0] Batch [20]	Speed: 113.19 samples/sec
INFO:root:Iter[0] Batch [20]	Speed: 111.20 samples/sec
INFO:root:Iter[0] Batch [20]	Speed: 110.38 samples/sec
INFO:root:Iter[0] Batch [25]	Speed: 111.24 samples/sec
INFO:root:Iter[0] Batch [25]	Speed: 109.90 samples/sec
INFO:root:Iter[0] Batch [25]	Speed: 107.48 samples/sec
```

`dist_aync`

```
INFO:root:Iter[0] Batch [5]	Speed: 202.15 samples/sec
INFO:root:Iter[0] Batch [5]	Speed: 181.41 samples/sec
INFO:root:Iter[0] Batch [5]	Speed: 179.61 samples/sec
INFO:root:Iter[0] Batch [10]	Speed: 125.75 samples/sec
INFO:root:Iter[0] Batch [10]	Speed: 108.90 samples/sec
INFO:root:Iter[0] Batch [10]	Speed: 109.25 samples/sec
INFO:root:Iter[0] Batch [15]	Speed: 118.44 samples/sec
INFO:root:Iter[0] Batch [15]	Speed: 112.89 samples/sec
INFO:root:Iter[0] Batch [15]	Speed: 112.83 samples/sec
INFO:root:Iter[0] Batch [20]	Speed: 123.68 samples/sec
INFO:root:Iter[0] Batch [20]	Speed: 115.85 samples/sec
INFO:root:Iter[0] Batch [20]	Speed: 105.82 samples/sec
INFO:root:Iter[0] Batch [25]	Speed: 124.24 samples/sec
INFO:root:Iter[0] Batch [25]	Speed: 115.21 samples/sec
INFO:root:Iter[0] Batch [25]	Speed: 106.60 samples/sec
INFO:root:Iter[0] Batch [30]	Speed: 120.62 samples/sec
INFO:root:Iter[0] Batch [30]	Speed: 121.35 samples/sec
```
