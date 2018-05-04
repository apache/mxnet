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
# Dockerfile to build MXNet on Ubuntu 16.04 for GPU but on
# a CPU-only instance. This restriction is caused by the CPP-
# package generation, requiring the actual CUDA library to be
# present

FROM nvidia/cuda:9.1-cudnn7-devel

ARG USER_ID=0

WORKDIR /work/deps

COPY install/ubuntu_core.sh /work/
RUN /work/ubuntu_core.sh
COPY install/ubuntu_python.sh /work/
RUN /work/ubuntu_python.sh
COPY install/ubuntu_scala.sh /work/
RUN /work/ubuntu_scala.sh
COPY install/ubuntu_r.sh /work/
RUN /work/ubuntu_r.sh
COPY install/ubuntu_perl.sh /work/
RUN /work/ubuntu_perl.sh
COPY install/ubuntu_clang.sh /work/
RUN /work/ubuntu_clang.sh
COPY install/ubuntu_mklml.sh /work/
RUN /work/ubuntu_mklml.sh

# Special case because the CPP-Package requires the CUDA runtime libs
# and not only stubs (which are provided by the base image)
COPY install/ubuntu_nvidia.sh /work/
RUN /work/ubuntu_nvidia.sh

# Keep this at the end since this command is not cachable
COPY install/ubuntu_adduser.sh /work/
RUN /work/ubuntu_adduser.sh

COPY runtime_functions.sh /work/

WORKDIR /work/mxnet
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib
