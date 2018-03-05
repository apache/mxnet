#!/bin/bash

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

clean_repo() {
    set -ex
    git clean -xfd
    git submodule foreach --recursive git clean -xfd
    git reset --hard
    git submodule foreach --recursive git reset --hard
    git submodule update --init --recursive
}


# Build commands: Every platform in docker/Dockerfile.build.<platform> should have a corresponding
# function here with the same suffix:

build_jetson() {
    set -ex
    pushd .
    cd /work/build
    cmake\
        -DUSE_CUDA=OFF\
        -DUSE_OPENCV=OFF\
        -DUSE_OPENMP=ON\
        -DUSE_SIGNAL_HANDLER=ON\
        -DUSE_MKL_IF_AVAILABLE=OFF\
        -DUSE_LAPACK=OFF\
        -DCMAKE_BUILD_TYPE=RelWithDebInfo\
        -G Ninja /work/mxnet
    ninja
    export MXNET_LIBRARY_PATH=`pwd`/libmxnet.so
    cd /work/mxnet/python
    python setup.py bdist_wheel --universal
    cp dist/*.whl /work/build
    popd
}

build_armv7() {
    set -ex
    pushd .
    cd /work/build
    cmake\
        -DUSE_CUDA=OFF\
        -DUSE_OPENCV=OFF\
        -DUSE_OPENMP=OFF\
        -DUSE_SIGNAL_HANDLER=ON\
        -DCMAKE_BUILD_TYPE=RelWithDebInfo\
        -DUSE_MKL_IF_AVAILABLE=OFF\
        -G Ninja /work/mxnet
    ninja
    export MXNET_LIBRARY_PATH=`pwd`/libmxnet.so
    cd /work/mxnet/python
    python setup.py bdist_wheel --universal
    cp dist/*.whl /work/build
    popd
}

build_ubuntu_cpu() {
    # TODO: Check where this is used
    set -ex
    pushd .
    cd /work/build
    cmake\
        -DUSE_CPP_PACKAGE=ON\
        -DUSE_CUDA=OFF\
        -DUSE_OPENCV=ON\
        -DUSE_OPENMP=ON\
        -DUSE_SIGNAL_HANDLER=ON\
        -DUSE_MKL_IF_AVAILABLE=OFF\
        -DCMAKE_BUILD_TYPE=RelWithDebInfo\
        -G Ninja /work/mxnet
    ninja
    popd
}

test_ubuntu_cpu_python2() {
    set -ex
    pushd .
    export MXNET_LIBRARY_PATH=/work/build/libmxnet.so

    VENV=mxnet_py2_venv
    virtualenv -p `which python2` $VENV
    source $VENV/bin/activate
    pip install nose nose-timer

    cd /work/mxnet/python
    pip install -e .
    cd /work/mxnet
    python -m "nose" --with-timer --verbose tests/python/unittest
    popd
}

test_ubuntu_cpu_python3() {
    set -ex
    pushd .
    export MXNET_LIBRARY_PATH=/work/build/libmxnet.so
    VENV=mxnet_py3_venv
    virtualenv -p `which python3` $VENV
    source $VENV/bin/activate

    cd /work/mxnet/python
    pip3 install nose nose-timer
    pip3 install -e .
    cd /work/mxnet
    python3 -m "nose" --with-timer --verbose tests/python/unittest

    popd
}

build_amzn_linux_cpu() {
    cd /work/build
    cmake\
        -DUSE_CUDA=OFF\
        -DUSE_OPENCV=ON\
        -DUSE_OPENMP=ON\
        -DUSE_SIGNAL_HANDLER=ON\
        -DCMAKE_BUILD_TYPE=RelWithDebInfo\
        -DUSE_MKL_IF_AVAILABLE=OFF\
        -DUSE_LAPACK=OFF\
        -G Ninja /work/mxnet
    ninja
    export MXNET_LIBRARY_PATH=`pwd`/libmxnet.so
}

build_arm64() {
    cmake\
        -DUSE_CUDA=OFF\
        -DUSE_OPENCV=OFF\
        -DUSE_OPENMP=OFF\
        -DUSE_SIGNAL_HANDLER=ON\
        -DCMAKE_BUILD_TYPE=RelWithDebInfo\
        -DUSE_MKL_IF_AVAILABLE=OFF\
        -G Ninja /work/mxnet
    ninja
    export MXNET_LIBRARY_PATH=`pwd`/libmxnet.so
    cd /work/mxnet/python
    python setup.py bdist_wheel --universal
    cp dist/*.whl /work/build
}

build_android_arm64() {
    set -ex
    cd /work/build
    cmake\
        -DUSE_CUDA=OFF\
        -DUSE_OPENCV=OFF\
        -DUSE_OPENMP=OFF\
        -DUSE_SIGNAL_HANDLER=ON\
        -DCMAKE_BUILD_TYPE=RelWithDebInfo\
        -DUSE_MKL_IF_AVAILABLE=OFF\
        -G Ninja /work/mxnet
    ninja
    export MXNET_LIBRARY_PATH=`pwd`/libmxnet.so
    cd /work/mxnet/python
    python setup.py bdist_wheel --universal
    cp dist/*.whl /work/build


}

build_ubuntu_gpu() {
    # TODO
    set -ex
    echo FIXME
}

build_centos7_cpu() {
    set -ex
    cd /work/mxnet
    make \
        DEV=1 \
        USE_PROFILER=1 \
        USE_BLAS=openblas \
        -j$(nproc)
}

build_centos7_gpu() {
    set -ex
    cd /work/mxnet
    make \
        DEV=1 \
        USE_PROFILER=1 \
        USE_BLAS=openblas \
        USE_CUDA=1 \
        USE_CUDA_PATH=/usr/local/cuda \
        USE_CUDNN=1 \
        -j$(nproc)
}

test_centos7_cpu() {
    set -ex
    cd /work/mxnet
    python3.6 -m "nose" --with-timer --verbose tests/python/unittest
    python3.6 -m "nose" --with-timer --verbose tests/python/train
}

test_centos7_gpu() {
    set -ex
    cd /work/mxnet
    python3.6 -m "nose" --with-timer --verbose tests/python/gpu
}



build_ubuntu_cpu_openblas() {
    set -ex
    make \
        DEV=1                         \
        USE_PROFILER=1                \
        USE_CPP_PACKAGE=1             \
        USE_BLAS=openblas             \
        -j$(nproc)
}

build_ubuntu_cpu_clang39() {
    set -ex
    make \
        USE_PROFILER=1                \
        USE_CPP_PACKAGE=1             \
        USE_BLAS=openblas             \
        USE_OPENMP=0                  \
        CXX=clang++-3.9               \
        CC=clang-3.9                  \
        -j$(nproc)
}

build_ubuntu_cpu_clang50() {
    set -ex
    make \
        USE_PROFILER=1                \
        USE_CPP_PACKAGE=1             \
        USE_BLAS=openblas             \
        USE_OPENMP=1                  \
        CXX=clang++-5.0               \
        CC=clang-5.0                  \
        -j$(nproc)
}

build_ubuntu_cpu_mkldnn() {
    set -ex
    make  \
        DEV=1                         \
        USE_PROFILER=1                \
        USE_CPP_PACKAGE=1             \
        USE_BLAS=openblas             \
        USE_MKLDNN=1                  \
        -j$(nproc)
}

build_ubuntu_gpu_mkldnn() {
    set -ex
    make  \
        DEV=1                         \
        USE_PROFILER=1                \
        USE_CPP_PACKAGE=1             \
        USE_BLAS=openblas             \
        USE_MKLDNN=1                  \
        USE_CUDA=1                    \
        USE_CUDA_PATH=/usr/local/cuda \
        USE_CUDNN=1                   \
        -j$(nproc)
}

build_ubuntu_gpu_cuda8_cudnn5() {
    set -ex
    make  \
        DEV=1                         \
        USE_PROFILER=1                \
        USE_BLAS=openblas             \
        USE_CUDA=1                    \
        USE_CUDA_PATH=/usr/local/cuda \
        USE_CUDNN=1                   \
        USE_CPP_PACKAGE=1             \
        -j$(nproc)
}

build_ubuntu_amalgamation() {
    set -ex
    make \
        amalgamation/                 \
        USE_BLAS=openblas             \
        -j$(nproc)
}

build_ubuntu_amalgamation_min() {
    set -ex
    make \
        amalgamation/                 \
        USE_BLAS=openblas             \
        MIN=1                         \
        -j$(nproc)
}

build_ubuntu_gpu_cmake_mkldnn() {
    set -ex
    cmake \
        -DUSE_CUDA=1               \
        -DUSE_CUDNN=1              \
        -DUSE_MKLML_MKL=1          \
        -DUSE_MKLDNN=1             \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja ..                \
    
    ninja -v
}

build_ubuntu_gpu_cmake() {
    set -ex
    cmake \
        -DUSE_CUDA=1               \
        -DUSE_CUDNN=1              \
        -DUSE_MKLML_MKL=0          \
        -DUSE_MKLDNN=0             \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja ..                \
    
    ninja -v
}

# Testing

sanity_check() {
    set -ex
    tools/license_header.py check
    make cpplint rcpplint jnilint
    make pylint
}


##############################################################
# MAIN
#
# Run function passed as argument
set +x
if [ $# -gt 0 ]
then
    $@
else
    cat<<EOF

$0: Execute a function by passing it as an argument to the script:

Possible commands:

EOF
    declare -F | cut -d' ' -f3
    echo
fi
