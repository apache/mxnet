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
# Python MXNet Dockerfile

# NOTE: Assumes wheel_build directory is the context root when building

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PYTHON=python3
ARG PIP=pip3
ARG PYTHON_VERSION=3.7.9
RUN apt-get update && \
    apt-get install -y wget ${PYTHON}-dev gcc && \
    wget -q https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar -xzf Python-$PYTHON_VERSION.tgz && \
    cd Python-$PYTHON_VERSION && \
    ./configure --enable-shared --prefix=/usr/local && \
    make -j $(nproc) && make install && \
    cd .. && rm -rf ../Python-$PYTHON_VERSION* && \
    ln -s /usr/local/bin/pip3 /usr/bin/pip && \
    ln -s /usr/local/bin/$PYTHON /usr/local/bin/python && \
    ${PIP} --no-cache-dir install --upgrade pip setuptools

ARG MXNET_COMMIT_ID
ENV MXNET_COMMIT_ID=${MXNET_COMMIT_ID}

RUN mkdir -p /mxnet
COPY dist/*.whl /mxnet/.

WORKDIR /mxnet
RUN WHEEL_FILE=$(ls -t /mxnet | head -n 1) && pip install ${WHEEL_FILE} && rm -f ${WHEEL_FILE}

