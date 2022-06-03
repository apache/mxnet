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

set -e

VARIANT=$1
DEPS_PATH=$2

source /etc/os-release

>&2 echo "Setting CUDA versions for $VARIANT"
if [[ $VARIANT == cu114* ]]; then
    CUDA_VERSION='11.4.67-1'
    CUDA_PATCH_VERSION='11.4.1.1026-1'
    CUDA_LIBS_VERSION='10.2.3.135-1'
    CUDA_SOLVER_VERSION='11.1.0.135-1'
    LIBCUDA_VERSION='460.32.03-0ubuntu1'
    LIBCUDNN_VERSION='8.1.1.33-1+cuda11.2'
    LIBNCCL_VERSION='2.8.3-1+cuda11.2'
    LIBCUDART_VERSION='11.2.72-1'
    LIBCUFFT_VERSION='10.4.0.135-1'
elif [[ $VARIANT == cu112* ]]; then
    CUDA_VERSION='11.2.67-1'
    CUDA_PATCH_VERSION='11.4.1.1026-1'
    CUDA_LIBS_VERSION='10.2.3.135-1'
    CUDA_SOLVER_VERSION='11.1.0.135-1'
    LIBCUDA_VERSION='460.32.03-0ubuntu1'
    LIBCUDNN_VERSION='8.1.1.33-1+cuda11.2'
    LIBNCCL_VERSION='2.8.3-1+cuda11.2'
    LIBCUDART_VERSION='11.2.72-1'
    LIBCUFFT_VERSION='10.4.0.135-1'
elif [[ $VARIANT == cu110* ]]; then
    CUDA_VERSION='11.0.221-1'
    CUDA_PATCH_VERSION='11.2.0.252-1'
    CUDA_LIBS_VERSION='10.2.1.245-1'
    CUDA_SOLVER_VERSION='10.6.0.245-1'
    CUDA_NVTX_VERSION='11.0.167-1'
    LIBCUDA_VERSION='450.36.06-0ubuntu1'
    LIBCUDNN_VERSION='8.0.4.30-1+cuda11.0'
    LIBNCCL_VERSION='2.7.8-1+cuda11.0'
elif [[ $VARIANT == cu102* ]]; then
    CUDA_VERSION='10.2.89-1'
    CUDA_PATCH_VERSION='10.2.2.89-1'
    LIBCUDA_VERSION='440.33.01-0ubuntu1'
    LIBCUDNN_VERSION='8.2.4.15-1+cuda10.2'
    LIBNCCL_VERSION='2.11.4-1+cuda10.2'
elif [[ $VARIANT == cu101* ]]; then
    CUDA_VERSION='10.1.105-1'
    CUDA_PATCH_VERSION='10.1.0.105-1'
    LIBCUDA_VERSION='418.39-0ubuntu1'
    LIBCUDNN_VERSION='7.6.5.32-1+cuda10.1'
    LIBNCCL_VERSION='2.5.6-1+cuda10.1'
elif [[ $VARIANT == cu100* ]]; then
    CUDA_VERSION='10.0.130-1'
    CUDA_PATCH_VERSION='10.0.130-1'
    LIBCUDA_VERSION='410.48-0ubuntu1'
    LIBCUDNN_VERSION='7.6.5.32-1+cuda10.0'
    LIBNCCL_VERSION='2.5.6-1+cuda10.0'
fi
if [[ $VARIANT == cu* ]]; then
    CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | tr '-' '.' | cut -d. -f1,2)
    CUDA_MAJOR_DASH=$(echo $CUDA_VERSION | tr '-' '.' | cut -d. -f1,2 | tr '.' '-')
    CUDA_PATCH_MAJOR_DASH=$(echo $CUDA_PATCH_VERSION | tr '-' '.' | cut -d. -f1,2 | tr '.' '-')
    NVIDIA_MAJOR_VERSION=$(echo $LIBCUDA_VERSION | cut -d. -f1)
    LIBCUDA_MAJOR=$(echo $LIBCUDA_VERSION | cut -d. -f1)
    LIBCUDNN_MAJOR=$(echo $LIBCUDNN_VERSION | cut -d. -f1)

    if [[ "$ID" == "centos" ]]; then
        distro="rhel${VERSION_ID}"
        arch=$(uname -m)
        sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/$distro/$(uname -m)/cuda-$distro.repo
    elif [[ "$ID" == "ubuntu" ]]; then
        distro=$(echo ${ID}${VERSION_ID} | sed 's/\.//g')
        wget -O /tmp/cuda.deb https://developer.download.nvidia.com/compute/cuda/repos/$distro/$(uname -m)/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i /tmp/cuda.deb
    fi

    export PATH=/usr/lib/binutils-2.26/bin/:${PATH}
    export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:$DEPS_PATH/usr/include
    export C_INCLUDE_PATH=${C_INCLUDE_PATH}:$DEPS_PATH/usr/include
    export LIBRARY_PATH=${LIBRARY_PATH}:$DEPS_PATH/usr/lib/x86_64-linux-gnu:$DEPS_PATH/usr/lib/nvidia-$NVIDIA_MAJOR_VERSION
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$DEPS_PATH/usr/lib/x86_64-linux-gnu:$DEPS_PATH/usr/lib/nvidia-$NVIDIA_MAJOR_VERSION
    export NVCC=/usr/local/cuda-$CUDA_MAJOR_VERSION/bin/nvcc

fi

# list of debs to download from nvidia
if [[ $VARIANT == cu* ]]; then
  if [[ $ID == "centos" ]]; then
    pkgs=(\
      "libcublas-$CUDA_MAJOR_DASH" \
      "libcublas-devel-$CUDA_MAJOR_DASH" \
      "cuda-cudart-$CUDA_MAJOR_DASH" \
      "cuda-cudart-devel-$CUDA_MAJOR_DASH" \
      "libcurand-$CUDA_MAJOR_DASH" \
      "libcurand-devel-$CUDA_MAJOR_DASH" \
      "libcufft-$CUDA_MAJOR_DASH" \
      "libcufft-devel-$CUDA_MAJOR_DASH" \
      "cuda-nvrtc-$CUDA_MAJOR_DASH" \
      "cuda-nvrtc-devel-$CUDA_MAJOR_DASH" \
      "libcusolver-$CUDA_MAJOR_DASH" \
      "libcusolver-devel-$CUDA_MAJOR_DASH" \
      "cuda-nvcc-$CUDA_MAJOR_DASH" \
      "cuda-nvtx-$CUDA_MAJOR_DASH" \
      "libcudnn${LIBCUDNN_MAJOR}-devel" \
      "cuda-libraries-${CUDA_MAJOR_DASH}" \
      "cuda-libraries-devel-${CUDA_MAJOR_DASH}" \
      "libnccl-devel" \
      "libnccl2" \
    )
  elif [[ $ID == "ubuntu" ]]; then
    pkgs=(\
      "libcublas-$CUDA_MAJOR_DASH" \
      "libcublas-dev-$CUDA_MAJOR_DASH" \
      "cuda-cudart-$CUDA_MAJOR_DASH" \
      "cuda-cudart-dev-$CUDA_MAJOR_DASH" \
      "libcurand-$CUDA_MAJOR_DASH" \
      "libcurand-dev-$CUDA_MAJOR_DASH" \
      "libcufft-$CUDA_MAJOR_DASH" \
      "libcufft-dev-$CUDA_MAJOR_DASH" \
      "cuda-nvrtc-$CUDA_MAJOR_DASH" \
      "cuda-nvrtc-dev-$CUDA_MAJOR_DASH" \
      "libcusolver-$CUDA_MAJOR_DASH" \
      "libcusolver-dev-$CUDA_MAJOR_DASH" \
      "cuda-nvcc-$CUDA_MAJOR_DASH" \
      "cuda-nvtx-$CUDA_MAJOR_DASH" \
      "libcudnn${LIBCUDNN_MAJOR}" \
      "libcudnn${LIBCUDNN_MAJOR}-dev" \
      "libnccl-dev" \
      "libnccl2" \
    )
  fi
fi

if [[ ! -d $DEPS_PATH/usr/local/cuda-${CUDA_MAJOR_VERSION} ]]; then
    prefix=$DEPS_PATH

    pkg_list=""
    for item in ${pkgs[*]}
    do
        pkg_list="$pkg_list ${item}"
    done

    if [[ "$ID" == "ubuntu" ]]; then
        sudo apt install -y $pkg_list
    elif [[ "$ID" == "centos" ]]; then
        sudo yum install -y $pkg_list
    fi
    #mkdir -p ${prefix}/include ${prefix}/lib ${prefix}/usr/lib/x86_64-linux-gnu
    #if [[ $LIBCUDNN_MAJOR == 8 ]]; then
    #    for h in ${prefix}/usr/include/x86_64-linux-gnu/cudnn_*_v8.h; do
    #        newfile=$(basename $h | sed 's/_v8//')
    #        cp $h ${prefix}/include/$newfile
    #    done
    #fi
    #cp -f ${prefix}/usr/include/x86_64-linux-gnu/cudnn_v${LIBCUDNN_MAJOR}.h ${prefix}/include/cudnn.h
    #ln -sf ${prefix}/usr/lib/x86_64-linux-gnu/libcudnn.so.${LIBCUDNN_MAJOR} ${prefix}/lib/libcudnn.so
    #cp -f ${prefix}/usr/include/nccl.h ${prefix}/include/nccl.h
fi

