# Performance

The following factors may significant affect the performance:

1. Use a fast backend. A fast BLAS library, e.g. openblas, altas,
and mkl, is necessary if only using CPU. While for Nvidia GPUs, we strongly
recommend to use CUDNN.
2. Three important things for the input data:
  1. data format. If you are using the `rec` format, then everything should be
    fine.
  2. decoding. In default MXNet uses 4 CPU threads for decoding the images, which
    are often able to decode over 1k images per second. You
    may increase the number of threads if either you are using a low-end CPU or
    you GPUs are very powerful.
  3. place to store the data. Any local or distributed filesystem (HDFS, Amazon
    S3) should be fine. There may be a problem if multiple machines read the
    data from the network shared filesystem (NFS) at the same time.
3. Use a large batch size. We often choose the largest one which can fit into
  the GPU memory. But a too large value may slow down the convergence. For
  example, the safe batch size for CIFAR 10 is around 200, while for ImageNet
  1K, the batch size can go beyond 1K.
4. Choose the proper `kvstore` if using more than one GPU. (See
  [doc/developer-guide/multi_node.md](../../doc/developer-guide/multi_node.md)
  for more information)
  1. For a single machine, often the default `local` is good enough. But you may want
  to use `local_allreduce_device` for models with size >> 100MB such as AlexNet
  and VGG. But also note that `local_allreduce_device` takes more GPU memory than
  others.
  2. For multiple machines, we recommend to try `dist_sync` first. But if the
  model size is quite large or you use a large number of machines, you may want to use `dist_async`.
