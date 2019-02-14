# -*- mode: dockerfile -*-
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#
# Dockerfile to run MXNet with TensorRT Integration.

FROM nvidia/cuda:9.0-cudnn7-devel

WORKDIR /work/deps

# Ubuntu-core
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
    libopenblas-base \
    python3 \
    python3-pip \
    wget

RUN ln -s $(which python3) /usr/bin/python && \
    ln -s $(which pip3) /usr/bin/pip

RUN echo "Installing TensorRT." && \
  wget -qO tensorrt.deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvinfer-runtime-trt-repo-ubuntu1604-4.0.1-ga-cuda9.0_1-1_amd64.deb && \
  dpkg -i tensorrt.deb && \
  apt-get update && \
  apt-get install -y --allow-downgrades libnvinfer-dev && \
  rm tensorrt.deb

RUN pip install mxnet-tensorrt-cu90

CMD python
