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

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.7-dev python3.7-distutils virtualenv wget && \
    ln -sf /usr/bin/python3.7 /usr/local/bin/python3 && \
    wget -nv https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py

RUN apt-get install -y libgomp1 libquadmath0

ARG MXNET_COMMIT_ID
ENV MXNET_COMMIT_ID=${MXNET_COMMIT_ID}

RUN mkdir -p /mxnet
COPY dist/*.whl /mxnet/.

WORKDIR /mxnet
RUN WHEEL_FILE=$(ls -t /mxnet | head -n 1) && pip install ${WHEEL_FILE} && rm -f ${WHEEL_FILE}

