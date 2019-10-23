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

ARG PYTHON_CMD=python
RUN apt-get update && \
    apt-get install -y wget ${PYTHON_CMD}-dev gcc && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    ${PYTHON_CMD} get-pip.py

ARG MXNET_COMMIT_ID
ENV MXNET_COMMIT_ID=${MXNET_COMMIT_ID}

RUN mkdir -p /mxnet
COPY dist/*.whl /mxnet/.

WORKDIR /mxnet
RUN WHEEL_FILE=$(ls -t /mxnet | head -n 1) && pip install ${WHEEL_FILE} && rm -f ${WHEEL_FILE}

