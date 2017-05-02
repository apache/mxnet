# -*- mode: dockerfile -*-
# dockerfile to build libmxnet.so on GPU
FROM nvidia/cuda:8.0-cudnn5-devel

COPY install/cpp.sh install/
RUN install/cpp.sh

ENV BUILD_OPTS "USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1"
RUN git clone --recursive https://github.com/dmlc/mxnet && cd mxnet && \
    make -j$(nproc) $BUILD_OPTS
