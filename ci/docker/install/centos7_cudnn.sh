#!/usr/bin/env bash

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

# build and install are separated so changes to build don't invalidate
# the whole docker cache for the image

set -ex

if [ -z ${CUDA_VERSION} ]; then
    echo "Error: CUDA_VERSION environment variable undefined"
    exit 1
fi

if [ -z ${CUDNN_VERSION} ]; then
    echo "Error: CUDNN_VERSION environment variable undefined"
    exit 1
fi

SHORT_CUDA_VERSION=""
SHORT_CUDNN_VERSION=""

if [[ ${CUDA_VERSION} =~ ([0-9]+\.[0-9]+)\.* ]]; then
    SHORT_CUDA_VERSION=${BASH_REMATCH[1]}
else
    echo "Error: CUDA_VERSION (${CUDA_VERSION}) did not match expected format [0-9]+.[0-9]+.*"
fi

if [[ ${CUDNN_VERSION} =~ ([0-9]+\.[0-9]+\.[0-9]+)\.* ]]; then
    SHORT_CUDNN_VERSION=${BASH_REMATCH[1]}
else
    echo "Error: CUDNN_VERSION (${CUDNN_VERSION}) did not match expected format [0-9]+.[0-9]+.[0-9]+.*"
fi

# Multipackage installation does not fail in yum
CUDNN_PKG="cudnn-${SHORT_CUDA_VERSION}-linux-x64-v${CUDNN_VERSION}.tgz"
CUDNN_PKG_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v${SHORT_CUDNN_VERSION}/${CUDNN_PKG}"
CUDNN_DOWNLOAD_SUM=`curl -fsSL "${CUDNN_PKG_URL}.sha256"`

curl -fsSL ${CUDNN_PKG_URL} -O
echo "${CUDNN_DOWNLOAD_SUM}" | sha256sum -c -
tar --no-same-owner -xzf ${CUDNN_PKG} -C /usr/local
rm ${CUDNN_PKG}
ldconfig
