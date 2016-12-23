# Performance

Here are some tips to get the most out of MXNet.

- [Intel CPU](#intel-cpu)
- [Nvidia GPU](#nvidia-gpu)
- [Multiple devices](#)

## Intel CPU

When using CPUs for training and inference, we suggest to use `MKL2017` by
enabling both `USE_MKL2017 = 1` and `USE_MKL2017_EXPERIMENTAL = 1` in
`config.mk`. Check
[MKL_README.md](https://github.com/dmlc/mxnet/blob/master/MKL_README.md) for
details

Also setting the following two environment variables may help:
- `KMP_AFFINITY=granularity=fine,compact,1,0` if there are two physical CPUs
- `OMP_NUM_THREADS=vCPUs / 2` in which `vGPUs` is the number of virtual CPUs,
  on linux we can get it by `cat /proc/cpuinfo  | grep processor | wc -l`

The following table shows the scoring performance, namely number of images can
be predicted per second. We used AWS EC2 C4.8xlarge (dual Intel(R) Xeon(R) CPU
E5-2666 v3 @ 2.90GHz) and
[example/image-classification/benchmark_score.py](https://github.com/dmlc/mxnet/blob/master/example/image-classification/benchmark_score.py)
with commit `0a03417` ## Nvidia GPU

| Batch | Alexnet | VGG | Inception-BN | Inception-v3 | Resnet 50 | Resnet 152 |
| --- | --- | --- | --- | --- | --- | --- |
|   1 |  98.441 | 24.66 |  27.43 |  11.91 |  38.74 | 14.95 |
|   2 | 121.676 | 27.07 |  24.78 |  11.61 |  44.27 | 15.62 |
|   4 | 254.394 | 38.51 |  28.76 |  12.86 |  40.88 | 19.01 |
|   8 | 237.733 | 36.97 |  25.57 |  12.68 |  43.00 | 16.11 |
|  16 | 280.822 | 40.00 |  20.85 |  11.77 |  55.00 | 16.93 |
|  32 | 285.417 | 44.40 |  31.03 |  12.45 |  55.70 | 17.02 |

## Multiple Devices

## Data

For the input data, mind the following:

* Data format. If you are using the `rec` format, then everything should be fine.
* Decoding. By default, MXNet uses 4 CPU threads for decoding images. This is often sufficient to decode more than 1K images per second. If  you are using a low-end CPU oryour GPUs are very powerful, you can increase the number of threads.
* Storage location. Any local or distributed file system (HDFS, Amazon S3) should be fine. If multiple devices read the data from the network shared file system (NFS) at the same time, problems might occur.
* Use a large batch size. We often choose the largest one that fits into GPU memory. A value that's too large can slow down convergence. For example, the safe batch size for CIFAR 10 is approximately 200, while for ImageNet 1K, the batch size can exceed 1K.

## Backend
* Use a fast BLAS library: e.g., openblas, atlas, or MKL. This is necessary only if you are using a CPU processor. If you are using Nvidia GPUs, we strongly
recommend using CUDNN.
* If you are using more than one GPU, choose the proper `kvstore`. For more information, see
  [doc/developer-guide/multi_node.md](http://mxnet.io/how_to/model_parallel_lstm.html).
* For a single device, the default `local` is usually good enough. For models greater than 100 MB, such as AlexNet
  and VGG, you might want
  to use `local_allreduce_device`. `local_allreduce_device` takes more GPU memory than
  other settings.
* For multiple devices, try using `dist_sync` first. If the
  size of the model is quite large or if you use a large number of devices, you might want to use `dist_async`.
