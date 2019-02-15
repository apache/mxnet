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
FROM ubuntu:14.04
MAINTAINER Mu Li <muli@cs.cmu.edu>

#
# First, build MXNet binaries (ref mxnet/docker/cpu/Dockerfile)
#

RUN apt-get update && apt-get install -y build-essential git libopenblas-dev liblapack-dev libopencv-dev
RUN git clone --recursive https://github.com/dmlc/mxnet/ && cd mxnet && \
    cp make/config.mk . && \
    echo "USE_BLAS=openblas" >>config.mk && \
    make -j$(nproc)

# python pakcage
RUN apt-get install -y python-numpy wget unzip
ENV PYTHONPATH /mxnet/python

#
# Now set up tools for doc build
#

RUN apt-get update && apt-get install -y \
    doxygen \
    build-essential \
    git \
    python-pip

RUN pip install sphinx==1.3.5 CommonMark==0.5.4 breathe mock==1.0.1 recommonmark

WORKDIR /opt/mxnet/docs

# Fool it into thinking it's on a READTHEDOCS server, so it builds the
# API reference
ENV READTHEDOCS true

ENTRYPOINT /opt/mxnet/docs/build-preview.sh

EXPOSE 8008

# Put this at the end so that you don't have to rebuild the earlier
# layers when iterating on the docs themselves.
ADD . /opt/mxnet/docs

