#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    >&2 echo "Usage: source set_environment.sh <VARIANT>[CPU|MKL|CU75|CU80|CU90|CU75MKL|CU80MKL|CU90MKL]"
fi
echo $PWD
export DEPS_PATH=$PWD/deps
export VARIANT=$(echo $1 | tr '[:upper:]' '[:lower:]')
export PLATFORM=$(uname | tr '[:upper:]' '[:lower:]')

# Variant-specific dependencies:
if [[ $PLATFORM == 'linux' ]]; then
    >&2 echo "Setting CUDA versions for $VARIANT"
    # TODO uncomment when starting to use mkldnn
    # if [[ $VARIANT == 'mkl' ]]; then
        # export MKLML_VERSION='2017.0.2.20170209'
        # export MKLDNN_VERSION='0.5'
    if [[ $VARIANT == cu90* ]]; then
        export CUDA_VERSION='9.0.176-1'
        export LIBCUDA_VERSION='384.66-0ubuntu1'
        # export LIBCUDNN_VERSION='6.0.21-1+cuda7.5'
        export LIBCUDNN_VERSION='7.0.4.31-1+cuda9.0'
    elif [[ $VARIANT == cu80* ]]; then
        export CUDA_VERSION='8.0.61-1'
        export LIBCUDA_VERSION='375.88-0ubuntu1'
        export LIBCUDNN_VERSION='7.0.4.31-1+cuda8.0'
    elif [[ $VARIANT == cu75* ]]; then
        export CUDA_VERSION='7.5-18'
        export LIBCUDA_VERSION='375.88-0ubuntu1'
        export LIBCUDNN_VERSION='6.0.21-1+cuda7.5'
    fi
    if [[ $VARIANT == cu* ]]; then
        # download and install cuda and cudnn, and set paths
        CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | tr '-' '.' | cut -d. -f1,2)
        NVIDIA_MAJOR_VERSION=$(echo $LIBCUDA_VERSION | cut -d. -f1)
        export PATH=${PATH}:$DEPS_PATH/usr/local/cuda-$CUDA_MAJOR_VERSION/bin
        export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:$DEPS_PATH/usr/local/cuda-$CUDA_MAJOR_VERSION/include
        export C_INCLUDE_PATH=${C_INCLUDE_PATH}:$DEPS_PATH/usr/local/cuda-$CUDA_MAJOR_VERSION/include
        export LIBRARY_PATH=${LIBRARY_PATH}:$DEPS_PATH/usr/local/cuda-$CUDA_MAJOR_VERSION/lib64:$DEPS_PATH/usr/lib/x86_64-linux-gnu:$DEPS_PATH/usr/lib/nvidia-$NVIDIA_MAJOR_VERSION
        export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$DEPS_PATH/usr/local/cuda-$CUDA_MAJOR_VERSION/lib64:$DEPS_PATH/usr/lib/x86_64-linux-gnu:$DEPS_PATH/usr/lib/nvidia-$NVIDIA_MAJOR_VERSION
    fi
fi
export PKG_CONFIG_PATH=$DEPS_PATH/lib/pkgconfig:$DEPS_PATH/lib64/pkgconfig:$PKG_CONFIG_PATH
export CPATH=$DEPS_PATH/include:$CPATH
export CC="gcc -fPIC"
export CXX="g++ -fPIC"
export FC="gfortran"
NUM_PROC=1
if [[ ! -z $(command -v nproc) ]]; then
    NUM_PROC=$(nproc)
elif [[ ! -z $(command -v sysctl) ]]; then
    NUM_PROC=$(sysctl -n hw.ncpu)
else
    >&2 echo "Can't discover number of cores."
fi
export NUM_PROC
>&2 echo "Using $NUM_PROC parallel jobs in building."
env
