# MXNET Builds

This directory contains the files and setup instructions to run all tests. They
are running on [ci.mxnet.io](http://ci.mxnet.io/blue/pipelines). But you can also
run them locally easily.

## Run locally

To run locally, we need to first install
[docker](https://docs.docker.com/engine/installation/) and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker/wiki).

Then we can run the tasks defined in the [Jenkinsfile](../../Jenkinsfile) by using (`ci_build.sh`)[./ci_build.sh]. For
example

- lint the python codes

  ```bash
  ./ci_build.sh lint make pylint
  ```

- build codes with CUDA supports

  ```bash
  ./ci_build.sh gpu make -j$(nproc) USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
  ```

- do the python unittest

  ```bash
  ./ci_build.sh gpu PYTHONPATH=./python/ nosetests --with-timer --verbose tests/python/unittest'
  ```
