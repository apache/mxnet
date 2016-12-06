FROM nvidia/cuda:7.5-cudnn5-devel
MAINTAINER Qingsong Liu <liuqs.ustc@gmail.com>

RUN apt-get update && apt-get install -y \
  git \
  libopenblas-dev \
  libopencv-dev \
  python-dev \
  python-numpy \
  python-setuptools \
  wget \
  python-pip \
  unzip

RUN cd /root && git clone --recursive https://github.com/dmlc/mxnet && cd mxnet && \
  cp make/config.mk . && \
  sed -i 's/USE_BLAS = atlas/USE_BLAS = openblas/g' config.mk && \
  sed -i 's/USE_CUDA = 0/USE_CUDA = 1/g' config.mk && \
  sed -i 's/USE_CUDA_PATH = NONE/USE_CUDA_PATH = \/usr\/local\/cuda/g' config.mk && \
  sed -i 's/USE_CUDNN = 0/USE_CUDNN = 1/g' config.mk && \
  make -j"$(nproc)"

ENV PYTHONPATH /root/mxnet/python

WORKDIR /root/mxnet
