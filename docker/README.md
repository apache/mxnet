# Docker images for MXNET

Pre-built docker images are available at https://hub.docker.com/r/dmlc/mxnet/

## How to use

1. First pull the pre-built image

   ```bash
   docker pull dmlc/mxnet
   ```
2. Then we can run the python shell in the docker

   ```bash
   docker run -ti dmlc/mxnet python
   ```
   For example
   ```bash
   $ docker run -ti dmlc/mxnet python
   Python 2.7.6 (default, Jun 22 2015, 17:58:13)
   [GCC 4.8.2] on linux2
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import mxnet as mx
   import mxnet as mx
   >>> quit()
   quit()
   ```

   Note: One may get the error message `libdc1394 error: Failed to initialize
   libdc1394`, which is due to opencv and can be ignored.

3. Train a model on MNIST to check everything works

   ```
   docker run dmlc/mxnet python /mxnet/example/image-classification/train_mnist.py
   ```

If the host machine has Nvidia CPUs, we can use `dmlc/mxnet:gpu`, which has both CUDA and CUDNN installed.
To launch the docker, we need to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) first.

1. Pull the image

   ```bash
   docker pull dmlc/mxnet:cuda
   ```

2. Train MNIST on GPU 0

   ```bash
   nvidia-docker run dmlc/mxnet:cuda python /mxnet/example/image-classification/train_mnist.py --gpus 0
   ```

## How to build

```bash
docker build -t dmlc/mxnet:cpu cpu
docker build -t dmlc/mxnet:cuda cuda
```
