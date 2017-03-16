# MXNET Builds

This directory contains the files and setup instructions to run all tests. They
are running on [ci.mxnet.io](http://ci.mxnet.io/blue/pipelines). But you also
run them locally easiliy.

## Run locally

To run these jobs locally, we need to first install
[docker](https://docs.docker.com/engine/installation/) and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker/wiki).

Then we can run the tasked defined in the [Jenkinsfile](../../Jenkinsfile). For
example, lint the python codes

```bash
./ci_build.sh lint make pylint
```

or build codes with CUDA supports

```bash
./ci_build.sh gpu make -j$(nproc) USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
```

and then do the unittest

```bash
./ci_build.sh gpu PYTHONPATH=./python/ nosetests --with-timer --verbose tests/python/unittest'
```
