# Some Tips for Improving MXNet Performance
Even after fixing the training or deployment environment and parallelization scheme,
a number of configuration settings and data-handling choices can impact the _MXNet_ performance.
In this document, we address some tips for improving _MXNet_ performance.

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

For using Intel Xeon CPUs for training and inference, we suggest enabling
both `USE_MKL2017 = 1` and `USE_MKL2017_EXPERIMENTAL = 1` in
`config.mk`. Check
[MKL_README.md](https://github.com/dmlc/mxnet/blob/master/MKL_README.md) for
details.

We also find that setting the following two environment variables can help:
- `export KMP_AFFINITY=granularity=fine,compact,1,0` if there are two physical CPUs
- `export OMP_NUM_THREADS=vCPUs / 2` in which `vCPUs` is the number of virtual CPUs.
  Whe using Linux, we can access this information by running `cat /proc/cpuinfo  | grep processor | wc -l`

Note that _MXNet_ treats all CPUs on a single machine as a single device.
So whether you specify `cpu(0)` or `cpu()`, _MXNet_ will use all CPU cores on the machine.

### Scoring results
The following table shows performance,
namely number of images that can be predicted per second.
We used [example/image-classification/benchmark_score.py](https://github.com/dmlc/mxnet/blob/master/example/image-classification/benchmark_score.py)
to measure the performance on different AWS EC2 machines.

AWS EC2 C4.8xlarge:

| Batch | Alexnet | VGG | Inception-BN | Inception-v3 | Resnet 50 | Resnet 152 |
| --- | --- | --- | --- | --- | --- | --- |
|   1 |  119.57 | 34.23 |  111.36 |  54.42 |  42.83 | 19.51 |
|   2 | 210.58 | 51.63 |  137.10 |  67.30 |  57.54 | 23.56 |
|   4 | 318.54 | 70.00 |  187.21 |  76.53 |  63.64 | 25.80 |
|   8 | 389.34 | 77.39 |  211.90 |  84.26 |  63.89 | 28.11 |
|  16 | 489.12 | 85.26 |  220.52 |  82.00 |  63.93 | 27.08 |
|  32 | 564.04 | 87.15 |  208.21 |  83.05 |  62.19 | 25.76 |

AWS EC2 C4.4xlarge:

| Batch | Alexnet | VGG | Inception-BN | Inception-v3 | Resnet 50 | Resnet 152 |
| --- | --- | --- | --- | --- | --- | --- |
|   1 |  109.96 | 23.00 |  71.82 |  28.10 |  30.66 | 11.81 |
|   2 | 124.56 | 24.86 |  81.61 |  31.32 |  32.73 | 12.82 |
|   4 | 157.01 | 26.60 |  86.77 |  32.94 |  33.32 | 13.16 |
|   8 | 178.40 | 30.67 |  88.58 |  33.52 |  33.32 | 13.32 |
|  16 | 189.52 | 35.61 |  90.36 |  33.63 |  32.94 | 13.18 |
|  32 | 196.61 | 38.98 |  105.27 |  33.77 |  32.65 | 13.00 |

AWS EC2 C4.2xlarge:

| Batch | Alexnet | VGG | Inception-BN | Inception-v3 | Resnet 50 | Resnet 152 |
| --- | --- | --- | --- | --- | --- | --- |
|   1 |  70.75 | 12.87 |  42.86 |  16.53 |  18.14 | 7.01 |
|   2 | 71.53 | 13.08 |  45.66 |  17.38 |  18.53 | 7.18 |
|   4 | 84.72 | 15.38 |  47.50 |  17.80 |  18.96 | 7.35 |
|   8 | 93.44 | 18.33 |  48.08 |  17.93 |  18.99 | 7.40 |
|  16 | 97.03 | 20.12 |  55.73 |  18.00 |  18.91 | 7.36 |
|  32 | 113.90 | 21.10 |  62.54 |  17.98 |  18.80 | 7.33 |

AWS EC2 C4.xlarge:

| Batch | Alexnet | VGG | Inception-BN | Inception-v3 | Resnet 50 | Resnet 152 |
| --- | --- | --- | --- | --- | --- | --- |
|   1 |  37.92 | 6.57 |  23.09 |  8.79 |  9.65 | 3.73 |
|   2 | 36.77 | 7.31 |  24.00 |  9.00 |  9.84 | 3.78 |
|   4 | 43.18 | 8.94 |  24.42 |  9.12 |  9.91 | 3.83 |
|   8 | 47.05 | 10.01 |  28.32 |  9.13 |  9.88 | 3.83 |
|  16 | 55.74 | 10.61 |  31.96 |  9.14 |  9.86 | 3.80 |
|  32 | 65.05 | 10.91 |  33.86 |  9.34 |  10.31 | 3.86 |

AWS EC2 C4.large:

| Batch | Alexnet | VGG | Inception-BN | Inception-v3 | Resnet 50 | Resnet 152 |
| --- | --- | --- | --- | --- | --- | --- |
|   1 |  19.86 | 3.67 |  12.20 |  4.59 |  5.11 | 1.97 |
|   2 | 19.37 | 4.24 |  12.41 |  4.64 |  5.15 | 1.98 |
|   4 | 22.64 | 4.89 |  14.34 |  4.66 |  5.16 | 2.00 |
|   8 | 27.19 | 5.25 |  16.17 |  4.66 |  5.16 | 1.99 |
|  16 | 31.82 | 5.46 |  17.24 |  4.76 |  5.35 | OOM |
|  32 | 34.67 | 5.55 |  17.64 |  4.88 |  OOM | OOM |

## Other CPU

If using CPUs (not just Intel CPUs -- ARMs also), NNPACK can improve the running performance with 2x~7x, please check [nnpack.md](./nnpack.md) for details.

## Nvidia GPU

`cuDNN` typically accelerates _MXNet_ performance on NVIDIA GPUs significantly,
especially for convolution layers.
We suggest always checking to make sure that a recent cuDNN version is used.

Setting the environment `export MXNET_CUDNN_AUTOTUNE_DEFAULT=1` sometimes also helps.

We show results when using various GPUs including K80 (EC2 p2.2xlarge), M40,
and P100 (DGX-1).

### Scoring results

Based on
[example/image-classification/benchmark_score.py](https://github.com/dmlc/mxnet/blob/master/example/image-classification/benchmark_score.py)
and MXNet commit `0a03417`, with cuDNN 5.1

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

  | Batch | Alexnet(\*8) | Inception-v3 | Resnet 50 |
  | --- | --- | --- | --- |
  |   1 | 230.69 | 9.81  | 13.83 |
  |   2 | 348.10 | 15.31 | 21.85 |
  |   4 | 457.28 | 20.48 | 29.58 |
  |   8 | 533.51 | 24.47 | 36.83 |
  |  16 | 582.36 | 28.46 | 43.60 |
  |  32 | 483.37 | 29.62 | 45.52 |

- M40

  | Batch | Alexnet(\*8) | Inception-v3 | Resnet 50 |
  | --- | --- | --- | --- |
  |   1 | 405.17  | 14.35 | 21.56 |
  |   2 | 606.32  | 23.96 | 36.48 |
  |   4 | 792.66  | 37.38 | 52.96 |
  |   8 | 1016.51 | 52.69 | 70.21 |
  |  16 | 1105.18 | 62.35 | 83.13 |
  |  32 | 1046.23 | 68.87 | 90.74 |

- P100

  | Batch | Alexnet(\*8) | Inception-v3 | Resnet 50 |
  | --- | --- | --- | --- |
  |   1 | 809.94  | 15.14  | 27.20  |
  |   2 | 1202.93 | 30.34  | 49.55  |
  |   4 | 1631.37 | 50.59  | 78.31  |
  |   8 | 1882.74 | 77.75  | 122.45 |
  |  16 | 2012.04 | 111.11 | 156.79 |
  |  32 | 1869.69 | 129.98 | 181.53 |

## Multiple Devices

If more than one GPU or machine are used, MXNet uses `kvstore` to communicate data.
It's critical to use the proper type of `kvstore` to get the best performance.
Refer to [multi_device.md](http://mxnet.io/how_to/multi_devices.html) for more
details.

Besides, we can use [tools/bandwidth](https://github.com/dmlc/mxnet/tree/master/tools/bandwidth)
to find the communication cost per batch.
Ideally, the communication cost should be less than the time to compute a batch.
To reduce the communication cost, we can consider:

- Exploring different `--kv-store` options.
- Increasing the batch size to improve the computation to communication ratio.

## Input Data

To make sure you're handling input data in a reasonable way consider the following:

* Data format: If you are using the `rec` format, then everything should be fine.
* Decoding: By default, _MXNet_ uses 4 CPU threads for decoding images.
This is often sufficient to decode more than 1K images per second.
If you are using a low-end CPU or your GPUs are very powerful, you can increase the number of threads.
* Storage location. Any local or distributed file system (HDFS, Amazon S3) should be fine.
If multiple devices read the data from the shared network file system (NFS) at the same time, problems might occur.
* Use a large batch size. We often choose the largest one that fits into GPU memory.
A value that's too large can slow down convergence.
For example, the safe batch size for CIFAR 10 is approximately 200, while for ImageNet 1K, the batch size can exceed 1K.

## Profiler

As of v0.9.1 (with the NNVM merge), _MXNet_ has a built-in profiler
that gives detailed information about execution time at the symbol level.
This feature complements general profiling tools like _nvprof_ and _gprof_
by summarizing at the operator level, instead of a function, kernel, or instruction level.

In order to be able to use the profiler, you must compile _MXNet_ with the `USE_PROFILER=1` flag in `config.mk`.

The profiler can then be turned on with an [environment variable](http://mxnet.io/how_to/env_var.html#control-the-profiler)
for an entire program run, or programmatically for just part of a run.
See [example/profiler](https://github.com/dmlc/mxnet/tree/master/example/profiler)
for complete examples of how to use the profiler in code, but briefly, the Python code looks like:

```
    mx.profiler.profiler_set_config(mode='all', filename='profile_output.json')
    mx.profiler.profiler_set_state('run')

    # Code to be profiled goes here...

    mx.profiler.profiler_set_state('stop')
```

The `mode` parameter can be set to

* `symbolic` to only include symbolic operations
* `all` to include all operations

After the program finishes, navigate to your browser's tracing (Example - chrome://tracing in a Chrome browser) and load the `profile_output.json` file output by the profiler to inspect the results.

![MLP Profile](https://cloud.githubusercontent.com/assets/17693755/18035938/0a43484a-6d93-11e6-80d4-241c6ca552ea.png)

Note that the output file can grow extremely large, so this approach is not recommended for general use.
