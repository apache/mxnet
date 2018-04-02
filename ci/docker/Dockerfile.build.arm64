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
# Dockerfile to build MXNet for ARM64/ARMv8

FROM dockcross/linux-arm64

ENV ARCH aarch64
ENV CC /usr/bin/aarch64-linux-gnu-gcc
ENV CXX /usr/bin/aarch64-linux-gnu-g++
ENV FC /usr/bin/aarch64-linux-gnu-gfortran-4.9
ENV HOSTCC gcc

WORKDIR /work

COPY install/arm64_openblas.sh /work/
RUN /work/arm64_openblas.sh

ENV LD_LIBRARY_PATH /opt/OpenBLAS/lib
ENV CPLUS_INCLUDE_PATH /opt/OpenBLAS/include
WORKDIR /work/mxnet

COPY runtime_functions.sh /work/