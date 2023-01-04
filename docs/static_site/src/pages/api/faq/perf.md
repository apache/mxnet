---
layout: page_category
title: Some Tips for Improving MXNet Performance
category: faq
faq_c: Speed
question: What are the best setup and data-handling tips and tricks for improving speed?
permalink: /api/faq/perf
---
<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->


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

When using Intel Xeon CPUs for training and inference, the `mxnet-mkl` package is recommended. Adding `--pre` installs a nightly build from master. Without it you will install the latest patched release of MXNet:

```
$ pip install mxnet-mkl [--pre]
```

Or build MXNet from source code with `USE_ONEDNN=1`. For Linux users, `USE_ONEDNN=1` will be turned on by default.

We also find that setting the following environment variables can help:


| Variable  | Description |
| :-------- | :---------- |
| `OMP_NUM_THREADS`            | Suggested value: `vCPUs / 2` in which `vCPUs` is the number of virtual CPUs. For more information, please see the guide for [setting the number of threads using an OpenMP environment variable](https://software.intel.com/en-us/mkl-windows-developer-guide-setting-the-number-of-threads-using-an-openmp-environment-variable) |
| `KMP_AFFINITY`               | Suggested value: `granularity=fine,compact,1,0`.  For more information, please see the guide for [Thread Affinity Interface (Linux* and Windows*)](https://software.intel.com/en-us/node/522691). |

Note that _MXNet_ treats all CPUs on a single machine as a single device.
So whether you specify `cpu(0)` or `cpu()`, _MXNet_ will use all CPU cores on the machine.

### Scoring results
The following table shows performance of MXNet-1.2.0.rc1,
namely number of images that can be predicted per second.
We used [example/image-classification/benchmark_score.py](https://github.com/apache/mxnet/blob/master/example/image-classification/benchmark_score.py)
to measure the performance on different AWS EC2 machines.

AWS EC2 C5.18xlarge:


| Batch | Alexnet | VGG 16    | Inception-BN | Inception-v3 | Resnet 50 | Resnet 152 |
|-------|---------|--------|--------------|--------------|-----------|------------|
| 1     | 390.53  | 81.57  | 124.13       | 62.26        | 76.22     | 32.92      |
| 2     | 596.45  | 100.84 | 206.58       | 93.36        | 119.55    | 46.80      |
| 4     | 710.77  | 119.04 | 275.55       | 127.86       | 148.62    | 59.36      |
| 8     | 921.40  | 120.38 | 380.82       | 157.11       | 167.95    | 70.78      |
| 16    | 1018.43 | 115.30 | 411.67       | 168.71       | 178.54    | 75.13      |
| 32    | 1290.31 | 107.19 | 483.34       | 179.38       | 193.47    | 85.86      |



AWS EC2 C5.9xlarge:


| Batch | Alexnet | VGG 16   | Inception-BN | Inception-v3 | Resnet 50 | Resnet 152 |
|-------|---------|-------|--------------|--------------|-----------|------------|
| 1     | 257.77  | 50.61 | 130.99       | 66.95        | 75.38     | 32.33      |
| 2     | 410.60  | 63.02 | 195.14       | 87.84        | 102.67    | 41.57      |
| 4     | 462.59  | 62.64 | 263.15       | 109.87       | 127.15    | 50.69      |
| 8     | 573.79  | 63.95 | 309.99       | 121.36       | 140.84    | 59.01      |
| 16    | 709.47  | 67.79 | 350.19       | 128.26       | 147.41    | 64.15      |
| 32    | 831.46  | 69.58 | 354.91       | 129.92       | 149.18    | 64.25      |


AWS EC2 C5.4xlarge:

| Batch | Alexnet | VGG 16   | Inception-BN | Inception-v3 | Resnet 50 | Resnet 152 |
|-------|---------|-------|--------------|--------------|-----------|------------|
| 1     | 214.15  | 29.32 | 114.97       | 47.96        | 61.01     | 23.92      |
| 2     | 310.04  | 34.81 | 150.09       | 60.89        | 71.16     | 27.92      |
| 4     | 330.69  | 34.56 | 186.63       | 74.15        | 86.86     | 34.37      |
| 8     | 378.88  | 35.46 | 204.89       | 77.05        | 91.10     | 36.93      |
| 16    | 424.00  | 36.49 | 211.55       | 78.39        | 91.23     | 37.34      |
| 32    | 481.95  | 37.23 | 213.71       | 78.23        | 91.68     | 37.26      |


AWS EC2 C5.2xlarge:

| Batch | Alexnet | VGG 16   | Inception-BN | Inception-v3 | Resnet 50 | Resnet 152 |
|-------|---------|-------|--------------|--------------|-----------|------------|
| 1     | 131.01  | 15.67 | 78.75        | 31.12        | 37.30     | 14.75      |
| 2     | 182.29  | 18.01 | 98.59        | 39.13        | 45.98     | 17.84      |
| 4     | 189.31  | 18.25 | 110.26       | 41.35        | 49.21     | 19.32      |
| 8     | 211.75  | 18.57 | 115.46       | 42.53        | 49.98     | 19.81      |
| 16    | 236.06  | 19.11 | 117.18       | 42.59        | 50.20     | 19.92      |
| 32    | 261.13  | 19.46 | 116.20       | 42.72        | 49.95     | 19.80      |


AWS EC2 C5.xlarge:

| Batch | Alexnet | VGG 16  | Inception-BN | Inception-v3 | Resnet 50 | Resnet 152 |
|-------|---------|------|--------------|--------------|-----------|------------|
| 1     | 36.64   | 3.93 | 27.06        | 10.09        | 12.98     | 5.06       |
| 2     | 49.21   | 4.49 | 29.67        | 10.80        | 12.94     | 5.14       |
| 4     | 50.12   | 4.50 | 30.31        | 10.83        | 13.17     | 5.19       |
| 8     | 54.71   | 4.58 | 30.22        | 10.89        | 13.19     | 5.20       |
| 16    | 60.23   | 4.70 | 30.20        | 10.91        | 13.23     | 5.19       |
| 32    | 66.37   | 4.76 | 30.10        | 10.90        | 13.22     | 5.15       |


## Other CPU

If using CPUs (not just Intel CPUs -- ARMs also), NNPACK can improve the running performance with 2x~7x, please check [nnpack.md](nnpack) for details.

## Nvidia GPU

`cuDNN` typically accelerates _MXNet_ performance on NVIDIA GPUs significantly,
especially for convolution layers.
We suggest always checking to make sure that a recent cuDNN version is used.

Setting the environment `export MXNET_CUDNN_AUTOTUNE_DEFAULT=1` sometimes also helps.

We show results when using various GPUs including K80 (EC2 p2.2xlarge), M60 (EC2 g3.4xlarge),
and V100 (EC2 p3.2xlarge).

### Scoring results

Based on
[example/image-classification/benchmark_score.py](https://github.com/apache/mxnet/blob/master/example/image-classification/benchmark_score.py)
and  MXNet-1.2.0.rc1, with cuDNN 7.0.5

- K80 (single GPU)

| Batch | Alexnet | VGG 16    | Inception-BN | Inception-v3 | Resnet 50 | Resnet 152 |
|-------|---------|--------|--------------|--------------|-----------|------------|
| 1     | 243.93  | 43.59  | 68.62        | 35.52        | 67.41     | 23.65      |
| 2     | 338.16  | 49.14  | 113.41       | 56.29        | 93.35     | 33.88      |
| 4     | 478.92  | 53.44  | 159.61       | 74.43        | 119.18    | 45.23      |
| 8     | 683.52  | 70.50  | 190.49       | 86.23        | 131.32    | 50.54      |
| 16    | 1004.66 | 109.01 | 254.20       | 105.70       | 155.40    | 62.55      |
| 32    | 1238.55 | 114.98 | 285.49       | 116.79       | 159.42    | 64.99      |
| 64 | 1346.72 | 123.56 | 308.73 | 122.21 | 167.58 | 70.21 |
| 128 | 1416.91 | OOM | 320.98 | 123.11 | 171.55 | 71.85 |
| 256 | 1462.97 | OOM | 329.16 | 127.53 | 153.01 | 57.23 |

- M60

| Batch | Alexnet | VGG 16    | Inception-BN | Inception-v3 | Resnet 50 | Resnet 152 |
|-------|---------|--------|--------------|--------------|-----------|------------|
| 1     | 243.49  | 59.95  | 101.97       | 48.30        | 95.46     | 39.29      |
| 2     | 491.04  | 69.14  | 170.35       | 80.27        | 142.61    | 60.17      |
| 4     | 711.54  | 78.94  | 257.89       | 123.09       | 182.36    | 76.51      |
| 8     | 1077.73 | 109.34 | 343.42       | 152.82       | 208.74    | 87.27      |
| 16    | 1447.21 | 144.93 | 390.25       | 166.32       | 220.73    | 92.41      |
| 32    | 1797.66 | 151.86 | 416.69       | 176.56       | 230.19    | 97.03      |
| 64 | 1779.38 | 150.18 | 427.51 | 183.47 | 239.12 | 101.59 |
| 128 | 1787.36 | OOM | 439.04 | 185.29 | 243.31 | 103.39 |
| 256 | 1899.10 | OOM | 450.22 | 183.42 | 242.36 | 100.98 |


- V100

| Batch | Alexnet | VGG 16    | Inception-BN | Inception-v3 | Resnet 50 | Resnet 152 |
|-------|---------|--------|--------------|--------------|-----------|------------|
| 1     | 659.51  | 205.16 | 157.37 | 87.71 | 162.15    | 61.38      |
| 2     | 1248.21 | 265.40 | 297.34 | 159.24 | 293.74    | 116.30     |
| 4     | 2122.41 | 333.97 | 520.91 | 279.84 | 479.14    | 195.17     |
| 8     | 3894.30 | 420.26 | 898.09 | 455.03 | 699.39    | 294.19     |
| 16    | 5815.58 | 654.16 | 1430.97 | 672.54 | 947.45    | 398.79     |
| 32    | 7906.09 | 708.43 | 1847.26 | 814.59 | 1076.81   | 451.82     |
| 64 | 9486.26 | 701.59 | 2134.89 | 899.01 | 1168.37 | 480.44 |
| 128 | 10177.84 | 703.30 | 2318.32 | 904.33 | 1233.15 | 511.79 |
| 256 | 10990.46 | 473.62 | 2425.28 | 960.20 | 1155.07 | 449.35 |

Below is the performance result on V100 using float 16.

| Batch | VGG 16  | Inception-BN | Inception-v3 | Resnet 50 | Resnet 152 |
| ----- | ------- | ------------ | ------------ | --------- | ---------- |
| 1     | 276.29  | 155.53       | 150.99       | 270.89    | 96.79      |
| 2     | 476.91  | 296.45       | 282.02       | 493.99    | 176.88     |
| 4     | 711.92  | 525.05       | 492.45       | 851.15    | 321.52     |
| 8     | 1047.11 | 900.26       | 807.94       | 1282.36   | 517.66     |
| 16    | 1299.88 | 1441.41      | 1192.21      | 1722.97   | 724.57     |
| 32    | 1486.63 | 1854.30      | 1512.08      | 2085.51   | 887.34     |
| 64    | 1219.65 | 2138.61      | 1687.35      | 2341.67   | 1002.90    |
| 128   | 1169.81 | 2317.39      | 1818.26      | 2355.04   | 1046.98    |
| 256   | 764.16  | 2425.16      | 1653.74      | 1991.88   | 976.73     |

### Training results

Based on
[example/image-classification/train_imagenet.py](https://github.com/apache/mxnet/blob/master/example/image-classification/train_imagenet.py)
and  MXNet-1.2.0.rc1, with CUDNN 7.0.5. The benchmark script is available at
[here](https://github.com/mli/mxnet-benchmark/blob/master/run_vary_batch.sh),
where the batch size for Alexnet is increased by 16x.

- K80 (single GPU)

| Batch | Alexnet(\*16) | Inception-v3 | Resnet 50 |
| --- | --- | --- | --- |
|   1 | 300.30 | 10.48 | 15.61 |
|   2 | 406.08 | 16.00 | 23.88 |
|   4 | 461.01 | 22.10 | 32.26 |
|   8 | 484.00 | 26.80 | 39.42 |
|  16 | 490.45 | 31.62 | 46.69 |
|  32 | 414.72 | 33.78 | 49.48 |

- M60

| Batch | Alexnet(\*16) | Inception-v3 | Resnet 50 |
| --- | --- | --- | --- |
|   1 | 380.96 | 14.06 | 20.55 |
|   2 | 530.53 | 21.90 | 32.65 |
|   4 | 600.17 | 31.96 | 45.57 |
|   8 | 633.60 | 40.58 | 54.92 |
|  16 | 639.37 | 46.88 | 64.44 |
|  32 | 576.54 | 50.05 | 68.34 |

- V100

| Batch | Alexnet(\*16) | Inception-v3 | Resnet 50 |
| --- | --- | --- | --- |
|   1 | 1629.52 | 21.83 | 34.54 |
|   2 | 2359.73 | 40.11 | 65.01 |
|   4 | 2687.89 | 72.79 | 113.49 |
|   8 | 2919.02 | 118.43 | 174.81 |
|  16 | 2994.32 | 173.15 | 251.22 |
|  32 | 2585.61 | 214.48 | 298.51 |
| 64 | 1984.21 | 247.43 | 343.19 |
| 128 | OOM | 253.68 | 363.69 |

## Multiple Devices

If more than one GPU or machine are used, MXNet uses `kvstore` to communicate data.
It's critical to use the proper type of `kvstore` to get the best performance.
Refer to [Distributed Training](https://mxnet.apache.org/api/faq/distributed_training.html) for more
details.

Besides, we can use [tools/bandwidth](https://github.com/apache/mxnet/tree/master/tools/bandwidth)
to find the communication cost per batch.
Ideally, the communication cost should be less than the time to compute a batch.
To reduce the communication cost, we can consider:

- Exploring different `--kv-store` options.
- Increasing the batch size to improve the computation to communication ratio.

Finally, MXNet is integrated with other distributed training frameworks, including [horovod](https://github.com/apache/mxnet/tree/master/example/distributed_training-horovod) and [BytePS](https://github.com/bytedance/byteps#use-byteps-in-your-code).

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

_MXNet_ has a built-in profiler
that gives detailed information about execution time at the operator level.
This feature complements general profiling tools like _nvprof_ and _gprof_
by summarizing at the operator level, instead of a function, kernel, or instruction level.

The profiler can be turned on with an [environment variable]({{'/api/faq/env_var#control-the-profiler' | relative_url}})
for an entire program run, or programmatically for just part of a run. Note that by default the profiler hides the details of each individual operator, and you can reveal the details by setting environment variables `MXNET_EXEC_BULK_EXEC_INFERENCE`, `MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN` and `MXNET_EXEC_BULK_EXEC_TRAIN` to 0.
See [example/profiler](https://github.com/apache/mxnet/tree/master/example/profiler)
for complete examples of how to use the profiler in code, or [this tutorial](https://mxnet.apache.org/api/python/docs/tutorials/performance/backend/profiler.html) on how to profile MXNet performance.

Briefly, the Python code looks like:

```python
    # wait for previous operations to complete
    mx.nd.waitall() 
    mx.profiler.set_config(profile_all=True, aggregate_stats=True, filename='profile_output.json')
    mx.profiler.set_state('run')

    # Code to be profiled goes here...

    # wait for previous operations to complete
    mx.nd.waitall() 
    mx.profiler.set_state('stop')
```

After the program finishes, navigate to your browser's tracing (Example - chrome://tracing in a Chrome browser) and load the `profile_output.json` file output by the profiler to inspect the results.

![MLP Profile](https://cloud.githubusercontent.com/assets/17693755/18035938/0a43484a-6d93-11e6-80d4-241c6ca552ea.png)

Note that the output file can grow extremely large, so this approach is not recommended for general use.
