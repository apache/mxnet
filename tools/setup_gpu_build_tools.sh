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

# This script installs the tools and libraries for CUDA GPU on Ubuntu.
# Usage: VARIANT=cu92mkl; DEPS_PATH=$HOME; setup_gpu_build_tools.sh $VARIANT $DEPS_PATH;
# It installs the tools into DEPS_PATH as specified by the second argument, and will set
# the following environment variables:
# PATH, CPLUS_INCLUDE_PATH, C_INCLUDE_PATH, LIBRARY_PATH, LD_LIBRARY_PATH, NVCC

VARIANT=$1
DEPS_PATH=$2

>&2 echo "Setting CUDA versions for $VARIANT"
if [[ $VARIANT == cu100* ]]; then
    CUDA_VERSION='10.0.130-1'
    CUDA_PATCH_VERSION='10.0.130-1'
    LIBCUDA_VERSION='410.48-0ubuntu1'
    LIBCUDNN_VERSION='7.3.1.20-1+cuda10.0'
    LIBNCCL_VERSION='2.3.4-1+cuda9.2'
elif [[ $VARIANT == cu92* ]]; then
    CUDA_VERSION='9.2.148-1'
    CUDA_PATCH_VERSION='9.2.148.1-1'
    LIBCUDA_VERSION='396.44-0ubuntu1'
    LIBCUDNN_VERSION='7.3.1.20-1+cuda9.2'
    LIBNCCL_VERSION='2.3.4-1+cuda9.2'
elif [[ $VARIANT == cu91* ]]; then
    CUDA_VERSION='9.1.85-1'
    CUDA_PATCH_VERSION='9.1.85.3-1'
    LIBCUDA_VERSION='396.44-0ubuntu1'
    LIBCUDNN_VERSION='7.1.3.16-1+cuda9.1'
    LIBNCCL_VERSION='2.2.12-1+cuda9.1'
elif [[ $VARIANT == cu90* ]]; then
    CUDA_VERSION='9.0.176-1'
    CUDA_PATCH_VERSION='9.0.176.3-1'
    LIBCUDA_VERSION='384.145-0ubuntu1'
    LIBCUDNN_VERSION='7.3.1.20-1+cuda9.0'
    LIBNCCL_VERSION='2.3.4-1+cuda9.0'
elif [[ $VARIANT == cu80* ]]; then
    CUDA_VERSION='8.0.61-1'
    CUDA_PATCH_VERSION='8.0.61.2-1'
    LIBCUDA_VERSION='375.88-0ubuntu1'
    LIBCUDNN_VERSION='7.2.1.38-1+cuda8.0'
    LIBNCCL_VERSION='2.3.4-1+cuda8.0'
elif [[ $VARIANT == cu75* ]]; then
    CUDA_VERSION='7.5-18'
    CUDA_PATCH_VERSION='7.5-18'
    LIBCUDA_VERSION='375.88-0ubuntu1'
    LIBCUDNN_VERSION='6.0.21-1+cuda7.5'
    LIBNCCL_VERSION=''
fi
if [[ $VARIANT == cu* ]]; then
    CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | tr '-' '.' | cut -d. -f1,2)
    CUDA_MAJOR_DASH=$(echo $CUDA_VERSION | tr '-' '.' | cut -d. -f1,2 | tr '.' '-')
    NVIDIA_MAJOR_VERSION=$(echo $LIBCUDA_VERSION | cut -d. -f1)
    LIBCUDA_MAJOR=$(echo $LIBCUDA_VERSION | cut -d. -f1)
    LIBCUDNN_MAJOR=$(echo $LIBCUDNN_VERSION | cut -d. -f1)
    os_name=$(cat /etc/*release | grep '^ID=' | sed 's/^.*=//g')
    os_version=$(cat /etc/*release | grep VERSION_ID | sed 's/^.*"\([0-9]*\)\.\([0-9]*\)"/\1\2/g')
    os_id="${os_name}${os_version}"
    if [[ $CUDA_MAJOR_DASH == 9-* ]] || [[ $CUDA_MAJOR_DASH == 10-* ]]; then
        os_id="ubuntu1604"
    fi
    export PATH=/usr/lib/binutils-2.26/bin/:${PATH}:$DEPS_PATH/usr/local/cuda-$CUDA_MAJOR_VERSION/bin
    export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:$DEPS_PATH/usr/local/cuda-$CUDA_MAJOR_VERSION/include
    export C_INCLUDE_PATH=${C_INCLUDE_PATH}:$DEPS_PATH/usr/local/cuda-$CUDA_MAJOR_VERSION/include
    export LIBRARY_PATH=${LIBRARY_PATH}:$DEPS_PATH/usr/local/cuda-$CUDA_MAJOR_VERSION/lib64:$DEPS_PATH/usr/lib/x86_64-linux-gnu:$DEPS_PATH/usr/lib/nvidia-$NVIDIA_MAJOR_VERSION
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$DEPS_PATH/usr/local/cuda-$CUDA_MAJOR_VERSION/lib64:$DEPS_PATH/usr/lib/x86_64-linux-gnu:$DEPS_PATH/usr/lib/nvidia-$NVIDIA_MAJOR_VERSION
    export NVCC=$DEPS_PATH/usr/local/cuda-$CUDA_MAJOR_VERSION/bin/nvcc
fi

# list of debs to download from nvidia
if [[ $VARIANT == cu100* ]]; then
    cuda_files=( \
      "cuda-core-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cublas-${CUDA_MAJOR_DASH}_${CUDA_PATCH_VERSION}_amd64.deb" \
      "cuda-cublas-dev-${CUDA_MAJOR_DASH}_${CUDA_PATCH_VERSION}_amd64.deb" \
      "cuda-cudart-${CUDA_MAJOR_DASH}_${CUDA_PATCH_VERSION}_amd64.deb" \
      "cuda-cudart-dev-${CUDA_MAJOR_DASH}_${CUDA_PATCH_VERSION}_amd64.deb" \
      "cuda-curand-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-curand-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cufft-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cufft-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-nvrtc-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-nvrtc-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cusolver-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cusolver-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-misc-headers-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-nvcc-${CUDA_MAJOR_DASH}_${CUDA_PATCH_VERSION}_amd64.deb" \
      "libcuda1-${LIBCUDA_MAJOR}_${LIBCUDA_VERSION}_amd64.deb" \
      "nvidia-${LIBCUDA_MAJOR}_${LIBCUDA_VERSION}_amd64.deb" \
    )
    ml_files=( \
      "libcudnn${LIBCUDNN_MAJOR}-dev_${LIBCUDNN_VERSION}_amd64.deb" \
      "libnccl-dev_${LIBNCCL_VERSION}_amd64.deb" \
    )
elif [[ $VARIANT == cu92* ]]; then
    cuda_files=( \
      "cuda-core-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cublas-${CUDA_MAJOR_DASH}_${CUDA_PATCH_VERSION}_amd64.deb" \
      "cuda-cublas-dev-${CUDA_MAJOR_DASH}_${CUDA_PATCH_VERSION}_amd64.deb" \
      "cuda-cudart-${CUDA_MAJOR_DASH}_${CUDA_PATCH_VERSION}_amd64.deb" \
      "cuda-cudart-dev-${CUDA_MAJOR_DASH}_${CUDA_PATCH_VERSION}_amd64.deb" \
      "cuda-curand-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-curand-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cufft-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cufft-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-nvrtc-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-nvrtc-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cusolver-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cusolver-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-misc-headers-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-nvcc-${CUDA_MAJOR_DASH}_${CUDA_PATCH_VERSION}_amd64.deb" \
      "libcuda1-${LIBCUDA_MAJOR}_${LIBCUDA_VERSION}_amd64.deb" \
      "nvidia-${LIBCUDA_MAJOR}_${LIBCUDA_VERSION}_amd64.deb" \
    )
    ml_files=( \
      "libcudnn${LIBCUDNN_MAJOR}-dev_${LIBCUDNN_VERSION}_amd64.deb" \
      "libnccl-dev_${LIBNCCL_VERSION}_amd64.deb" \
    )
elif [[ $VARIANT == cu91* ]]; then
    cuda_files=( \
      "cuda-core-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cublas-${CUDA_MAJOR_DASH}_${CUDA_PATCH_VERSION}_amd64.deb" \
      "cuda-cublas-dev-${CUDA_MAJOR_DASH}_${CUDA_PATCH_VERSION}_amd64.deb" \
      "cuda-cudart-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cudart-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-curand-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-curand-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cufft-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cufft-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-nvrtc-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-nvrtc-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cusolver-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cusolver-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-misc-headers-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-nvcc-${CUDA_MAJOR_DASH}_9.1.85.2-1_amd64.deb" \
      "libcuda1-${LIBCUDA_MAJOR}_${LIBCUDA_VERSION}_amd64.deb" \
      "nvidia-${LIBCUDA_MAJOR}_${LIBCUDA_VERSION}_amd64.deb" \
    )
    ml_files=( \
      "libcudnn${LIBCUDNN_MAJOR}-dev_${LIBCUDNN_VERSION}_amd64.deb" \
      "libnccl-dev_${LIBNCCL_VERSION}_amd64.deb" \
    )
elif [[ $VARIANT == cu90* ]]; then
    cuda_files=( \
      "cuda-core-${CUDA_MAJOR_DASH}_${CUDA_PATCH_VERSION}_amd64.deb" \
      "cuda-cublas-${CUDA_MAJOR_DASH}_9.0.176.4-1_amd64.deb" \
      "cuda-cublas-dev-${CUDA_MAJOR_DASH}_9.0.176.4-1_amd64.deb" \
      "cuda-cudart-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cudart-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-curand-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-curand-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cufft-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cufft-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-nvrtc-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-nvrtc-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cusolver-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cusolver-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-misc-headers-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "libcuda1-${LIBCUDA_MAJOR}_${LIBCUDA_VERSION}_amd64.deb" \
      "nvidia-${LIBCUDA_MAJOR}_${LIBCUDA_VERSION}_amd64.deb" \
    )
    ml_files=( \
      "libcudnn${LIBCUDNN_MAJOR}-dev_${LIBCUDNN_VERSION}_amd64.deb" \
      "libnccl-dev_${LIBNCCL_VERSION}_amd64.deb" \
    )
elif [[ $VARIANT == cu80* ]]; then
    cuda_files=( \
      "cuda-core-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cublas-${CUDA_MAJOR_DASH}_${CUDA_PATCH_VERSION}_amd64.deb" \
      "cuda-cublas-dev-${CUDA_MAJOR_DASH}_${CUDA_PATCH_VERSION}_amd64.deb" \
      "cuda-cudart-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cudart-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-curand-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-curand-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cufft-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cufft-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-nvrtc-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-nvrtc-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cusolver-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cusolver-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-misc-headers-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "libcuda1-${LIBCUDA_MAJOR}_${LIBCUDA_VERSION}_amd64.deb" \
      "nvidia-${LIBCUDA_MAJOR}_${LIBCUDA_VERSION}_amd64.deb" \
    )
    ml_files=( \
      "libcudnn${LIBCUDNN_MAJOR}-dev_${LIBCUDNN_VERSION}_amd64.deb" \
      "libnccl-dev_${LIBNCCL_VERSION}_amd64.deb" \
    )
elif [[ $VARIANT == cu75* ]]; then
    cuda_files=( \
      "cuda-core-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cublas-${CUDA_MAJOR_DASH}_${CUDA_PATCH_VERSION}_amd64.deb" \
      "cuda-cublas-dev-${CUDA_MAJOR_DASH}_${CUDA_PATCH_VERSION}_amd64.deb" \
      "cuda-cudart-${CUDA_MAJOR_DASH}_${CUDA_PATCH_VERSION}_amd64.deb" \
      "cuda-cudart-dev-${CUDA_MAJOR_DASH}_${CUDA_PATCH_VERSION}_amd64.deb" \
      "cuda-curand-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-curand-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cufft-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cufft-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-nvrtc-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-nvrtc-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cusolver-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-cusolver-dev-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "cuda-misc-headers-${CUDA_MAJOR_DASH}_${CUDA_VERSION}_amd64.deb" \
      "libcuda1-${LIBCUDA_MAJOR}_${LIBCUDA_VERSION}_amd64.deb" \
      "nvidia-${LIBCUDA_MAJOR}_${LIBCUDA_VERSION}_amd64.deb" \
    )
    ml_files=( \
      "libcudnn${LIBCUDNN_MAJOR}-dev_${LIBCUDNN_VERSION}_amd64.deb" \
    )
fi


if [[ ! -d $DEPS_PATH/usr/local/cuda-${CUDA_MAJOR_VERSION} ]]; then
    prefix=$DEPS_PATH

    for item in ${cuda_files[*]}
    do
        echo "Installing $item"
        curl -sL "http://developer.download.nvidia.com/compute/cuda/repos/${os_id}/x86_64/${item}" -o package.deb
        dpkg -X package.deb ${prefix}
        rm package.deb
    done
    for item in ${ml_files[*]}
    do
        echo "Installing $item"
        curl -sL "http://developer.download.nvidia.com/compute/machine-learning/repos/${os_id}/x86_64/${item}" -o package.deb
        dpkg -X package.deb ${prefix}
        rm package.deb
    done

    mkdir -p ${prefix}/include
    mkdir -p ${prefix}/lib
    cp ${prefix}/usr/include/x86_64-linux-gnu/cudnn_v${LIBCUDNN_MAJOR}.h ${prefix}/include/cudnn.h
    ln -s libcudnn_static_v${LIBCUDNN_MAJOR}.a ${prefix}/usr/lib/x86_64-linux-gnu/libcudnn.a
    cp ${prefix}/usr/local/cuda-${CUDA_MAJOR_VERSION}/lib64/*.a ${prefix}/lib/
    cp ${prefix}/usr/include/nccl.h ${prefix}/include/nccl.h
    ln -s libnccl_static.a ${prefix}/usr/lib/x86_64-linux-gnu/libnccl.a
fi
