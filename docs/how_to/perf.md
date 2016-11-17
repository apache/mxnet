# Performance

The following factors can significantly affect performance:

1. Use a fast back-end. A fast BLAS library, e.g., openblas, altas,
or mkl, is necessary only if you are using a CPU processor. If you are using Nvidia GPUs, we strongly
recommend using CUDNN.
2. For the input data, the following:
	* Data format. If you are using the `rec` format, then everything should be fine.
	* Decoding. By default, MXNet uses 4 CPU threads for decoding images. This is often sufficient to decode more than 1K images per second. If  you are using a low-end CPU or
    your GPUs are very powerful, you
    can increase the number of threads.
	* Storage location. Any local or distributed file system (HDFS, Amazon
    S3) should be fine. If multiple devices read the
    data from the network shared file system (NFS) at the same time, problems might occur.
3. Use a large batch size. We often choose the largest one that fits into
   GPU memory. A value that's too large can slow down convergence. For
  example, the safe batch size for CIFAR 10 is approximately 200, while for ImageNet
  1K, the batch size can exceed 1K.
4. If you are using more than one GPU, choose the proper `kvstore`. For more information, see
  [doc/developer-guide/multi_node.md](../../doc/developer-guide/multi_node.md).
	* For a single device, the default `local` is usually good enough. For models greater than 100 MB, such as AlexNet
  and VGG, you might want
  to use `local_allreduce_device`. `local_allreduce_device` takes more GPU memory than
  other settings.
	* For multiple devices, try using `dist_sync` first. If the
  size of the model is quite large or if you use a large number of devices, you might want to use `dist_async`.
