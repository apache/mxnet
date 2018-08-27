# -*- mode: dockerfile -*-
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
# Dockerfile to run MXNet on Ubuntu 16.04 for GPU

FROM nvidia/cuda:9.1-cudnn7-devel

WORKDIR /work/deps

COPY install/ubuntu_core.sh /work/
RUN /work/ubuntu_core.sh

COPY install/deb_ubuntu_ccache.sh /work/
RUN /work/deb_ubuntu_ccache.sh

COPY install/ubuntu_python.sh /work/
RUN /work/ubuntu_python.sh

COPY install/ubuntu_scala.sh /work/
COPY install/sbt.gpg /work/
RUN /work/ubuntu_scala.sh

COPY install/ubuntu_r.sh /work/
COPY install/r.gpg /work/
RUN /work/ubuntu_r.sh

COPY install/ubuntu_perl.sh /work/
RUN /work/ubuntu_perl.sh

COPY install/ubuntu_clang.sh /work/
RUN /work/ubuntu_clang.sh

COPY install/ubuntu_mklml.sh /work/
RUN /work/ubuntu_mklml.sh

COPY install/ubuntu_tvm.sh /work/
RUN /work/ubuntu_tvm.sh

COPY install/ubuntu_llvm.sh /work/
RUN /work/ubuntu_llvm.sh

COPY install/ubuntu_caffe.sh /work/
RUN /work/ubuntu_caffe.sh

COPY install/ubuntu_onnx.sh /work/
RUN /work/ubuntu_onnx.sh

COPY install/ubuntu_docs.sh /work/
COPY install/docs_requirements /work/
RUN /work/ubuntu_docs.sh

COPY install/ubuntu_tutorials.sh /work/
RUN /work/ubuntu_tutorials.sh

ARG USER_ID=0
ARG GROUP_ID=0
COPY install/ubuntu_adduser.sh /work/
RUN /work/ubuntu_adduser.sh

COPY runtime_functions.sh /work/

WORKDIR /work/mxnet
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib
