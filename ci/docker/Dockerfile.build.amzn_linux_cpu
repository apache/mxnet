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
# Dockerfile to build and run MXNet for Amazon Linux on CPU

FROM amazonlinux

WORKDIR /work/deps
COPY install/amzn_linux_core.sh /work/
RUN /work/amzn_linux_core.sh
COPY install/amzn_linux_opencv.sh /work/
RUN /work/amzn_linux_opencv.sh
COPY install/amzn_linux_openblas.sh /work/
RUN /work/amzn_linux_openblas.sh
COPY install/amzn_linux_python2.sh /work/
RUN /work/amzn_linux_python2.sh
COPY install/amzn_linux_python3.sh /work/
RUN /work/amzn_linux_python3.sh
COPY install/amzn_linux_testdeps.sh /work/
RUN /work/amzn_linux_testdeps.sh
COPY install/amzn_linux_julia.sh /work/
RUN /work/amzn_linux_julia.sh
COPY install/amzn_linux_maven.sh /work/
RUN /work/amzn_linux_maven.sh
COPY install/amzn_linux_library.sh /work/
RUN /work/amzn_linux_library.sh
WORKDIR /work/mxnet

COPY runtime_functions.sh /work/