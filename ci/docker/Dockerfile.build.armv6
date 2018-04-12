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
# Dockerfile to build MXNet for ARMv6

FROM dockcross/linux-armv6

ENV ARCH armv6l
ENV FC=/usr/bin/${CROSS_TRIPLE}-gfortran
ENV HOSTCC gcc
ENV TARGET ARMV6

WORKDIR /work/deps

# Build OpenBLAS
RUN git clone --recursive -b v0.2.20 https://github.com/xianyi/OpenBLAS.git && \
    cd OpenBLAS && \
    make -j$(nproc) && \
    make PREFIX=$CROSS_ROOT install

COPY runtime_functions.sh /work/
WORKDIR /work/mxnet
