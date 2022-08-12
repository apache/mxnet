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

# This script installs the Cuda build tools and libraries into
# the standard system paths.
# Usage: VARIANT=cu112; setup_gpu_build_tools.sh $VARIANT

set -e

VARIANT=$1
DEPS_PATH=$2

source /etc/os-release

>&2 echo "Setting CUDA versions for $VARIANT"
if [[ $VARIANT == cu117* ]]; then
    CUDA_VERSION='11.7'
    LIBCUDNN_VERSION='8.5.0.96'
elif [[ $VARIANT == cu116* ]]; then
    CUDA_VERSION='11.6'
    LIBCUDNN_VERSION='8.4.1.50'
elif [[ $VARIANT == cu115* ]]; then
    CUDA_VERSION='11.5'
    LIBCUDNN_VERSION='8.3.3.40'
elif [[ $VARIANT == cu114* ]]; then
    CUDA_VERSION='11.4'
    LIBCUDNN_VERSION='8.2.4.15'
elif [[ $VARIANT == cu113* ]]; then
    CUDA_VERSION='11.3'
    LIBCUDNN_VERSION='8.2.1.32'
elif [[ $VARIANT == cu112* ]]; then
    CUDA_VERSION='11.2'
    LIBCUDNN_VERSION='8.1.1.33'
elif [[ $VARIANT == cu111* ]]; then
    CUDA_VERSION='11.1'
    LIBCUDNN_VERSION='8.0.5.39'
elif [[ $VARIANT == cu110* ]]; then
    CUDA_VERSION='11.0'
    LIBCUDNN_VERSION='8.0.5.39'
else
    echo "Unsupported CUDA variant '$VARIANT'"
    exit -1
fi

CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | tr '-' '.' | cut -d. -f1,2)
CUDA_MAJOR_DASH=$(echo $CUDA_VERSION | tr '-' '.' | cut -d. -f1,2 | tr '.' '-')
LIBCUDNN_MAJOR=$(echo $LIBCUDNN_VERSION | cut -d. -f1)

if [[ "$ID" == "centos" ]]; then
    distro="rhel${VERSION_ID}"
    sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/$distro/$(uname -m)/cuda-$distro.repo
elif [[ "$ID" == "ubuntu" ]]; then
    distro=$(echo ${ID}${VERSION_ID} | sed 's/\.//g')
    wget -O /tmp/cuda.deb https://developer.download.nvidia.com/compute/cuda/repos/$distro/$(uname -m)/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i /tmp/cuda.deb
fi

if [[ $ID == "centos" ]]; then
    pkgs=(\
        "cuda-libraries-${CUDA_MAJOR_DASH}" \
        "cuda-libraries-devel-${CUDA_MAJOR_DASH}" \
        "cuda-nvcc-$CUDA_MAJOR_DASH" \
        "cuda-nvtx-$CUDA_MAJOR_DASH" \
        "cuda-nvprof-${CUDA_MAJOR_DASH}" \
        "libcudnn${LIBCUDNN_MAJOR}-${LIBCUDNN_VERSION}-1.cuda${CUDA_MAJOR_VERSION}" \
        "libcudnn${LIBCUDNN_MAJOR}-devel-${LIBCUDNN_VERSION}-1.cuda${CUDA_MAJOR_VERSION}" \
        "libnccl-devel" \
        "libnccl2" \
    )
elif [[ $ID == "ubuntu" ]]; then
    pkgs=(\
        "cuda-libraries-${CUDA_MAJOR_DASH}" \
        "cuda-libraries-dev-${CUDA_MAJOR_DASH}" \
        "cuda-nvcc-$CUDA_MAJOR_DASH" \
        "cuda-nvtx-$CUDA_MAJOR_DASH" \
        "cuda-nvprof-$CUDA_MAJOR_DASH" \
        "libcudnn${LIBCUDNN_MAJOR}-${LIBCUDNN_VERSION}-1+cuda${CUDA_MAJOR_VERSION}" \
        "libcudnn${LIBCUDNN_MAJOR}-dev-${LIBCUDNN_VERSION}-1+cuda${CUDA_MAJOR_VERSION}" \
        "libnccl-dev" \
        "libnccl2" \
    )
fi

if [[ ! -d /usr/local/cuda-${CUDA_MAJOR_VERSION} ]]; then

    if [[ "$ID" == "ubuntu" ]]; then
        sudo apt install -y ${pkgs[*]}
    elif [[ "$ID" == "centos" ]]; then
        sudo yum install -y ${pkgs[*]}
    fi

fi

# allow linking against libcuda stubs if no driver is present
export CMAKE_PARAMETERS="-DCMAKE_EXE_LINKER_FLAGS='-L/usr/local/cuda-${CUDA_MAJOR_VERSION}/lib64/stubs' \
	-DCMAKE_SHARED_LINKER_FLAGS='-L/usr/local/cuda-${CUDA_MAJOR_VERSION}/lib64/stubs'"

