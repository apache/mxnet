# Performance

Performance is mainly affected by the following 4 factors:

1. Implementation of operators (Convolution, Pooling, ..)
   - [Intel CPU](#intel-cpu)
   - [Nvidia GPU](#nvidia-gpu)
2. Input data loading and augmentation
   - [Input Data](#input-data)
3. Workloads (computation graph) optimization and scheduling
   - [Profiler](#profiler)
4. Communication for multi-devices training
   - [Multiple Devices](#multiple-devices)

## Intel CPU

For using Intel Xeon CPUs for training and inference, we suggest to enable
both `USE_MKL2017 = 1` and `USE_MKL2017_EXPERIMENTAL = 1` in
`config.mk`. Check
[MKL_README.md](https://github.com/dmlc/mxnet/blob/master/MKL_README.md) for
details

Also setting the following two environment variables may help:
- `KMP_AFFINITY=granularity=fine,compact,1,0` if there are two physical CPUs
- `OMP_NUM_THREADS=vCPUs / 2` in which `vCPUs` is the number of virtual CPUs.
  For linux we can get it by `cat /proc/cpuinfo  | grep processor | wc -l`

Note that MXNet treats all CPU in a single machine as a single device. So when
specify `cpu(0)` or `cpu()`, all CPU cores in the machine will be used.

### Scoring results
The following table shows the scoring performance, namely number of images can
be predicted per second. We used AWS EC2 C4.8xlarge (dual Intel(R) Xeon(R) CPU
E5-2666 v3 @ 2.90GHz) and
[example/image-classification/benchmark_score.py](https://github.com/dmlc/mxnet/blob/master/example/image-classification/benchmark_score.py)
with MXNet commit `0a03417`

| Batch | Alexnet | VGG | Inception-BN | Inception-v3 | Resnet 50 | Resnet 152 |
| --- | --- | --- | --- | --- | --- | --- |
|   1 |  98.44 | 24.66 |  27.43 |  11.91 |  38.74 | 14.95 |
|   2 | 121.67 | 27.07 |  24.78 |  11.61 |  44.27 | 15.62 |
|   4 | 254.39 | 38.51 |  28.76 |  12.86 |  40.88 | 19.01 |
|   8 | 237.73 | 36.97 |  25.57 |  12.68 |  43.00 | 16.11 |
|  16 | 280.82 | 40.00 |  20.85 |  11.77 |  55.00 | 16.93 |
|  32 | 285.41 | 44.40 |  31.03 |  12.45 |  55.70 | 17.02 |

## Nvidia GPU

`cuDNN` often greatly accelerate performance on Nvidia GPUs, especially for
convolution layers. Please check a recent CUDNN version is used.

Setting the environment `export MXNET_CUDNN_AUTOTUNE_DEFAULT=1` sometimes also helps.

We show performance results of various GPUs including K80 (EC2 p2.2xlarge), M40,
and P100 (DGX-1).

### Scoring results

Based on
[example/image-classification/benchmark_score.py](https://github.com/dmlc/mxnet/blob/master/example/image-classification/benchmark_score.py)
and MXNet commit `0a03417`, with CUDNN 5.1

- K80 (single GPU)

  | Batch | Alexnet | VGG | Inception-BN | Inception-v3 | Resnet 50 | Resnet 152 |
  | --- | --- | --- | --- | --- | --- | --- |
  |   1 | 202.66  | 70.76 | 74.91  | 42.61  | 70.94 | 24.87 |
  |   2 | 233.76  | 63.53 | 119.60  | 60.09  | 92.28 | 34.23 |
  |   4 | 367.91  | 78.16 | 164.41  | 72.30  | 116.68 | 44.76 |
  |   8 | 624.14  | 119.06 | 195.24  | 79.62  | 129.37 | 50.96 |
  |  16 | 1071.19 | 195.83 | 256.06  | 99.38  | 160.40 | 66.51 |
  |  32 | 1443.90 | 228.96 | 287.93  | 106.43  | 167.12 | 69.73 |

- M40

  | Batch | Alexnet | VGG | Inception-BN | Inception-v3 | Resnet 50 | Resnet 152 |
  | --- | --- | --- | --- | --- | --- | --- |
  |   1 | 412.09 | 142.10 | 115.89  | 64.40  | 126.90 | 46.15 |
  |   2 | 743.49 | 212.21 | 205.31  | 108.06  | 202.17 | 75.05 |
  |   4 | 1155.43 | 280.92 | 335.69  | 161.59  | 266.53 | 106.83 |
  |   8 | 1606.87 | 332.76 | 491.12  | 224.22  | 317.20 | 128.67 |
  |  16 | 2070.97 | 400.10 | 618.25  | 251.87  | 335.62 | 134.60 |
  |  32 | 2694.91 | 466.95 | 624.27  | 258.59  | 373.35 | 152.71 |

- P100

  | Batch | Alexnet | VGG | Inception-BN | Inception-v3 | Resnet 50 | Resnet 152 |
  | --- | --- | --- | --- | --- | --- | --- |
  |   1 | 624.84 | 294.6 | 139.82  | 80.17  | 162.27 | 58.99 |
  |   2 | 1226.85 | 282.3 | 267.41  | 142.63  | 278.02 | 102.95 |
  |   4 | 1934.97 | 399.3 | 463.38  | 225.56  | 423.63 | 168.91 |
  |   8 | 2900.54 | 522.9 | 709.30  | 319.52  | 529.34 | 210.10 |
  |  16 | 4063.70 | 755.3 | 949.22  | 444.65  | 647.43 | 270.07 |
  |  32 | 4883.77 | 854.4 | 1197.74  | 493.72  | 713.17 | 294.17 |

### Training results

Based on
[example/image-classification/train_imagenet.py](https://github.com/dmlc/mxnet/blob/master/example/image-classification/train_imagenet.py)
and MXNet commit `0a03417`, with CUDNN 5.1. The benchmark script is available at
[here](https://github.com/mli/mxnet-benchmark/blob/master/run_vary_batch.sh),
where the batch size for Alexnet is increased by 8x.

- K80 (single GPU)

  | Batch | Alexnet(*8) | Inception-v3 | Resnet 50 |
  | --- | --- | --- | --- |
  |   1 | 230.69 | 9.81  | 13.83 |
  |   2 | 348.10 | 15.31 | 21.85 |
  |   4 | 457.28 | 20.48 | 29.58 |
  |   8 | 533.51 | 24.47 | 36.83 |
  |  16 | 582.36 | 28.46 | 43.60 |
  |  32 | 483.37 | 29.62 | 45.52 |

- M40

  | Batch | Alexnet(*8) | Inception-v3 | Resnet 50 |
  | --- | --- | --- | --- |
  |   1 | 405.17  | 14.35 | 21.56 |
  |   2 | 606.32  | 23.96 | 36.48 |
  |   4 | 792.66  | 37.38 | 52.96 |
  |   8 | 1016.51 | 52.69 | 70.21 |
  |  16 | 1105.18 | 62.35 | 83.13 |
  |  32 | 1046.23 | 68.87 | 90.74 |

- P100

  | Batch | Alexnet(*8) | Inception-v3 | Resnet 50 |
  | --- | --- | --- | --- |
  |   1 | 809.94  | 15.14  | 27.20  |
  |   2 | 1202.93 | 30.34  | 49.55  |
  |   4 | 1631.37 | 50.59  | 78.31  |
  |   8 | 1882.74 | 77.75  | 122.45 |
  |  16 | 2012.04 | 111.11 | 156.79 |
  |  32 | 1869.69 | 129.98 | 181.53 |

## Multiple Devices

If more than one GPU or machine are used, MXNet uses `kvstore` to communicate
data. A proper type of `kvstore` is critical to get the best performance. We can
refer to [mutli_device.md](http://mxnet.io/how_to/multi_devices.html) for more
details.

Besides, we can use
[tools/bandwidth](https://github.com/dmlc/mxnet/tree/master/tools/bandwidth) to
find the communication cost per batch. A ideal situation is the cost is less
than the time to compute a batch. We can

- Explore different `--kv-store` options to reduce the cost
- Increase the batch size to improve the computation and communication ratio.

## Input Data

For the input data, mind the following:

* Data format. If you are using the `rec` format, then everything should be fine.
* Decoding. By default, MXNet uses 4 CPU threads for decoding images. This is often sufficient to decode more than 1K images per second. If  you are using a low-end CPU oryour GPUs are very powerful, you can increase the number of threads.
* Storage location. Any local or distributed file system (HDFS, Amazon S3) should be fine. If multiple devices read the data from the network shared file system (NFS) at the same time, problems might occur.
* Use a large batch size. We often choose the largest one that fits into GPU memory. A value that's too large can slow down convergence. For example, the safe batch size for CIFAR 10 is approximately 200, while for ImageNet 1K, the batch size can exceed 1K.

## Profiler

See [example/profiler](https://github.com/dmlc/mxnet/tree/nnvm/example/profiler)
on the nnvm branch.
