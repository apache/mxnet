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
#
# build and install are separated so changes to build don't invalidate
# the whole docker cache for the image

set -ex

NOSE_COVERAGE_ARGUMENTS="--with-coverage --cover-inclusive --cover-xml --cover-branches --cover-package=mxnet"
NOSE_TIMER_ARGUMENTS="--with-timer --timer-ok 1 --timer-warning 15 --timer-filter warning,error"
CI_CUDA_COMPUTE_CAPABILITIES="-gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_70,code=sm_70"
CI_CMAKE_CUDA_ARCH="5.2 7.0"

clean_repo() {
    set -ex
    git clean -xfd
    git submodule foreach --recursive git clean -xfd
    git reset --hard
    git submodule foreach --recursive git reset --hard
    git submodule update --init --recursive
}

scala_prepare() {
    # Clean up maven logs
    export MAVEN_OPTS="-Dorg.slf4j.simpleLogger.log.org.apache.maven.cli.transfer.Slf4jMavenTransferListener=warn"
}

check_cython() {
    set -ex
    local is_cython_used=$(python3 <<EOF
import sys
import mxnet as mx
cython_ndarraybase = 'mxnet._cy3.ndarray'
print(mx.nd._internal.NDArrayBase.__module__ == cython_ndarraybase)
EOF
)

    if [ "${is_cython_used}" != "True" ]; then
        echo "ERROR: cython is not used."
        return 1
    else
        echo "NOTE: cython is used."
        return 0
    fi
}

build_wheel() {

    set -ex
    pushd .

    PYTHON_DIR=${1:-/work/mxnet/python}
    BUILD_DIR=${2:-/work/build}

    # build

    export MXNET_LIBRARY_PATH=${BUILD_DIR}/libmxnet.so

    cd ${PYTHON_DIR}
    python3 setup.py bdist_wheel

    # repackage

    # Fix pathing issues in the wheel.  We need to move libmxnet.so from the data folder to the
    # mxnet folder, then repackage the wheel.
    WHEEL=`readlink -f dist/*.whl`
    TMPDIR=`mktemp -d`
    unzip -d ${TMPDIR} ${WHEEL}
    rm ${WHEEL}
    cd ${TMPDIR}
    mv *.data/data/mxnet/libmxnet.so mxnet
    zip -r ${WHEEL} .
    cp ${WHEEL} ${BUILD_DIR}
    rm -rf ${TMPDIR}

    popd
}

gather_licenses() {
    mkdir -p licenses

    cp tools/dependencies/LICENSE.binary.dependencies licenses/
    cp NOTICE licenses/
    cp LICENSE licenses/
    cp DISCLAIMER-WIP licenses/
}

build_ubuntu_cpu_release() {
    set -ex

    make  \
        DEV=0                         \
        ENABLE_TESTCOVERAGE=0         \
        USE_CPP_PACKAGE=0             \
        USE_MKLDNN=0                  \
        USE_BLAS=openblas             \
        USE_SIGNAL_HANDLER=1          \
        -j$(nproc)
}

build_ubuntu_cpu_mkldnn_release() {
    set -ex

    make  \
        DEV=0                         \
        ENABLE_TESTCOVERAGE=0         \
        USE_CPP_PACKAGE=0             \
        USE_MKLDNN=1                  \
        USE_BLAS=openblas             \
        USE_SIGNAL_HANDLER=1          \
        -j$(nproc)
}

build_ubuntu_gpu_release() {
    set -ex

    make \
        DEV=0                                     \
        ENABLE_TESTCOVERAGE=0                     \
        USE_BLAS=openblas                         \
        USE_MKLDNN=0                              \
        USE_CUDA=1                                \
        USE_CUDA_PATH=/usr/local/cuda             \
        USE_CUDNN=1                               \
        USE_CPP_PACKAGE=0                         \
        USE_DIST_KVSTORE=1                        \
        USE_SIGNAL_HANDLER=1                      \
        -j$(nproc)
}

build_ubuntu_gpu_mkldnn_release() {
    set -ex

    make \
        DEV=0                                     \
        ENABLE_TESTCOVERAGE=0                     \
        USE_BLAS=openblas                         \
        USE_MKLDNN=1                              \
        USE_CUDA=1                                \
        USE_CUDA_PATH=/usr/local/cuda             \
        USE_CUDNN=1                               \
        USE_CPP_PACKAGE=0                         \
        USE_DIST_KVSTORE=1                        \
        USE_SIGNAL_HANDLER=1                      \
        -j$(nproc)
}

# Compiles the dynamic mxnet library
# Parameters:
# $1 -> mxnet_variant: the mxnet variant to build, e.g. cpu, native, cu101, cu102, etc.
build_dynamic_libmxnet() {
    set -ex

    local mxnet_variant=${1:?"This function requires a mxnet variant as the first argument"}

    # relevant licenses will be placed in the licenses directory
    gather_licenses

    cd /work/build
    source /opt/rh/devtoolset-8/enable
    # Opt in to newer GCC C++ ABI. devtoolset defaults to ABI Version 2.
    export CXXFLAGS="-fabi-version=11 -fabi-compat-version=7"
    if [[ ${mxnet_variant} = "cpu" ]]; then
        cmake -DUSE_MKL_IF_AVAILABLE=OFF \
            -DUSE_MKLDNN=ON \
            -DUSE_CUDA=OFF \
            -G Ninja /work/mxnet
    elif [[ ${mxnet_variant} = "native" ]]; then
        cmake -DUSE_MKL_IF_AVAILABLE=OFF \
            -DUSE_MKLDNN=OFF \
            -DUSE_CUDA=OFF \
            -G Ninja /work/mxnet
    elif [[ ${mxnet_variant} =~ cu[0-9]+$ ]]; then
        cmake -DUSE_MKL_IF_AVAILABLE=OFF \
            -DUSE_MKLDNN=ON \
            -DUSE_DIST_KVSTORE=ON \
            -DUSE_CUDA=ON \
            -G Ninja /work/mxnet
    else
        echo "Error: Unrecognized mxnet variant '${mxnet_variant}'"
        exit 1
    fi
    ninja
}

build_jetson() {
    set -ex
    cd /work/build
    cmake \
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
        -DUSE_CUDA=ON \
        -DMXNET_CUDA_ARCH="5.2" \
        -DUSE_OPENCV=OFF \
        -DUSE_OPENMP=ON \
        -DUSE_LAPACK=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -G Ninja /work/mxnet
    ninja
    build_wheel
}

#
# ARM builds
#

build_armv6() {
    set -ex
    cd /work/build

    # We do not need OpenMP, since most armv6 systems have only 1 core

    cmake \
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
        -DUSE_CUDA=OFF \
        -DUSE_OPENCV=OFF \
        -DUSE_OPENMP=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_LAPACK=OFF \
        -DBUILD_CPP_EXAMPLES=OFF \
        -G Ninja /work/mxnet

    ninja
    build_wheel
}

build_armv7() {
    set -ex
    cd /work/build

    cmake \
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
        -DUSE_CUDA=OFF \
        -DUSE_OPENCV=OFF \
        -DUSE_OPENMP=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_LAPACK=OFF \
        -DBUILD_CPP_EXAMPLES=OFF \
        -G Ninja /work/mxnet

    ninja
    build_wheel
}

build_armv8() {
    cd /work/build
    cmake \
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
        -DUSE_CUDA=OFF \
        -DUSE_OPENCV=OFF \
        -DUSE_OPENMP=ON \
        -DUSE_LAPACK=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -G Ninja /work/mxnet
    ninja
    build_wheel
}


#
# ANDROID builds
#

build_android_armv7() {
    set -ex
    cd /work/build
    # ANDROID_ABI and ANDROID_STL are options of the CMAKE_TOOLCHAIN_FILE
    # provided by Android NDK
    cmake \
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
        -DANDROID_ABI="armeabi-v7a" \
        -DANDROID_STL="c++_shared" \
        -DUSE_CUDA=OFF \
        -DUSE_LAPACK=OFF \
        -DUSE_OPENCV=OFF \
        -DUSE_OPENMP=OFF \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -G Ninja /work/mxnet
    ninja
}

build_android_armv8() {
    set -ex
    cd /work/build
    # ANDROID_ABI and ANDROID_STL are options of the CMAKE_TOOLCHAIN_FILE
    # provided by Android NDK
    cmake \
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
        -DANDROID_ABI="arm64-v8a" \
        -DANDROID_STL="c++_shared" \
        -DUSE_CUDA=OFF \
        -DUSE_LAPACK=OFF \
        -DUSE_OPENCV=OFF \
        -DUSE_OPENMP=OFF \
        -DUSE_SIGNAL_HANDLER=ON \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -G Ninja /work/mxnet
    ninja
}

build_centos7_cpu() {
    set -ex
    cd /work/build
    source /opt/rh/devtoolset-7/enable
    # Opt in to newer GCC C++ ABI. devtoolset defaults to ABI Version 2.
    export CXXFLAGS="-fabi-version=11 -fabi-compat-version=7"
    cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_MKLDNN=OFF \
        -DUSE_DIST_KVSTORE=ON \
        -DUSE_CUDA=OFF \
        -DBUILD_EXTENSION_PATH=/work/mxnet/example/extensions/lib_external_ops \
        -DUSE_INT64_TENSOR_SIZE=OFF \
        -G Ninja /work/mxnet
    ninja
}

build_centos7_mkldnn() {
    set -ex
    cd /work/build
    source /opt/rh/devtoolset-7/enable
    # Opt in to newer GCC C++ ABI. devtoolset defaults to ABI Version 2.
    export CXXFLAGS="-fabi-version=11 -fabi-compat-version=7"
    cmake \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_MKLDNN=ON \
        -DUSE_CUDA=OFF \
        -DUSE_INT64_TENSOR_SIZE=OFF \
        -G Ninja /work/mxnet
    ninja
}

build_centos7_gpu() {
    set -ex
    cd /work/build
    source /opt/rh/devtoolset-7/enable
    # Opt in to newer GCC C++ ABI. devtoolset defaults to ABI Version 2.
    export CXXFLAGS="-fabi-version=11 -fabi-compat-version=7"
    cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_MKLDNN=ON \
        -DUSE_CUDA=ON \
        -DMXNET_CUDA_ARCH="$CI_CMAKE_CUDA_ARCH" \
        -DUSE_DIST_KVSTORE=ON \
        -DBUILD_EXTENSION_PATH=/work/mxnet/example/extensions/lib_external_ops \
        -DUSE_INT64_TENSOR_SIZE=OFF \
        -G Ninja /work/mxnet
    ninja
}

build_ubuntu_cpu() {
    build_ubuntu_cpu_openblas
}

build_ubuntu_cpu_openblas() {
    set -ex
    cd /work/build
    CXXFLAGS="-Wno-error=strict-overflow" CC=gcc-7 CXX=g++-7 cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DENABLE_TESTCOVERAGE=ON \
        -DUSE_TVM_OP=ON \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_MKLDNN=OFF \
        -DUSE_CUDA=OFF \
        -DUSE_DIST_KVSTORE=ON \
        -DBUILD_CYTHON_MODULES=ON \
        -DBUILD_EXTENSION_PATH=/work/mxnet/example/extensions/lib_external_ops \
        -G Ninja /work/mxnet
    ninja
}

build_ubuntu_cpu_mkl() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DENABLE_TESTCOVERAGE=OFF \
        -DUSE_MKLDNN=OFF \
        -DUSE_CUDA=OFF \
        -DUSE_TVM_OP=ON \
        -DUSE_MKL_IF_AVAILABLE=ON \
        -DUSE_MKL_LAYERNORM=ON \
        -DUSE_BLAS=MKL \
        -DBUILD_EXTENSION_PATH=/work/mxnet/example/extensions/lib_external_ops \
        -GNinja /work/mxnet
    ninja
}

build_ubuntu_cpu_cmake_debug() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DCMAKE_BUILD_TYPE=Debug \
        -DENABLE_TESTCOVERAGE=ON \
        -DUSE_CUDA=OFF \
        -DUSE_TVM_OP=ON \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_OPENMP=OFF \
        -DUSE_OPENCV=ON \
        -DUSE_SIGNAL_HANDLER=ON \
        -G Ninja \
        /work/mxnet
    ninja
}

build_ubuntu_cpu_cmake_no_tvm_op() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DUSE_CUDA=OFF \
        -DUSE_TVM_OP=OFF \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_OPENMP=OFF \
        -DUSE_OPENCV=ON \
        -DUSE_SIGNAL_HANDLER=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_EXTENSION_PATH=/work/mxnet/example/extensions/lib_external_ops \
        -G Ninja \
        /work/mxnet

    ninja
}

build_ubuntu_cpu_cmake_asan() {
    set -ex

    cd /work/build
    cmake \
        -DUSE_CUDA=OFF \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_MKLDNN=OFF \
        -DUSE_OPENMP=OFF \
        -DUSE_OPENCV=OFF \
        -DCMAKE_BUILD_TYPE=Debug \
        -DUSE_GPERFTOOLS=OFF \
        -DUSE_JEMALLOC=OFF \
        -DUSE_ASAN=ON \
        /work/mxnet
    make -j $(nproc) mxnet
}

build_ubuntu_cpu_clang6() {
    set -ex
    cd /work/build
    CXX=clang++-6.0 CC=clang-6.0 cmake \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_MKLDNN=OFF \
        -DUSE_CUDA=OFF \
        -DUSE_OPENMP=OFF \
        -DUSE_DIST_KVSTORE=ON \
        -G Ninja /work/mxnet
    ninja
}

build_ubuntu_cpu_clang100() {
    set -ex
    cd /work/build
    CXX=clang++-10 CC=clang-10 cmake \
       -DUSE_MKL_IF_AVAILABLE=OFF \
       -DUSE_MKLDNN=OFF \
       -DUSE_CUDA=OFF \
       -DUSE_OPENMP=ON \
       -DUSE_DIST_KVSTORE=ON \
       -G Ninja /work/mxnet
    ninja
}

build_ubuntu_cpu_clang_tidy() {
    set -ex
    cd /work/build
    # TODO(leezu) USE_OPENMP=OFF 3rdparty/dmlc-core/CMakeLists.txt:79 broken?
    CXX=clang++-10 CC=clang-10 cmake \
       -DUSE_MKL_IF_AVAILABLE=OFF \
       -DUSE_MKLDNN=OFF \
       -DUSE_CUDA=OFF \
       -DUSE_OPENMP=OFF \
       -DCMAKE_BUILD_TYPE=Debug \
       -DUSE_DIST_KVSTORE=ON \
       -DCMAKE_CXX_CLANG_TIDY=clang-tidy-6.0 \
       -G Ninja /work/mxnet
    ninja
}

build_ubuntu_cpu_clang6_mkldnn() {
    set -ex
    cd /work/build
    CXX=clang++-6.0 CC=clang-6.0 cmake \
       -DUSE_MKL_IF_AVAILABLE=OFF \
       -DUSE_MKLDNN=ON \
       -DUSE_CUDA=OFF \
       -DUSE_OPENMP=OFF \
       -G Ninja /work/mxnet
    ninja
}

build_ubuntu_cpu_clang100_mkldnn() {
    set -ex
    cd /work/build
    CXX=clang++-10 CC=clang-10 cmake \
       -DUSE_MKL_IF_AVAILABLE=OFF \
       -DUSE_MKLDNN=ON \
       -DUSE_CUDA=OFF \
       -G Ninja /work/mxnet
    ninja
}

build_ubuntu_cpu_mkldnn() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DENABLE_TESTCOVERAGE=ON \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_TVM_OP=ON \
        -DUSE_MKLDNN=ON \
        -DUSE_CUDA=OFF \
        -DBUILD_EXTENSION_PATH=/work/mxnet/example/extensions/lib_external_ops \
        -G Ninja /work/mxnet
    ninja
}

build_ubuntu_cpu_mkldnn_mkl() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DENABLE_TESTCOVERAGE=OFF \
        -DUSE_MKLDNN=ON \
        -DUSE_CUDA=OFF \
        -DUSE_TVM_OP=ON \
        -DUSE_MKL_IF_AVAILABLE=ON \
        -DUSE_BLAS=MKL \
        -DBUILD_EXTENSION_PATH=/work/mxnet/example/extensions/lib_external_ops \
        -GNinja /work/mxnet
    ninja
}

build_ubuntu_gpu_tensorrt() {

    set -ex

    export CC=gcc-7
    export CXX=g++-7
    export ONNX_NAMESPACE=onnx

    # Build ONNX
    pushd .
    echo "Installing ONNX."
    cd 3rdparty/onnx-tensorrt/third_party/onnx
    rm -rf build
    mkdir -p build
    cd build
    cmake -DCMAKE_CXX_FLAGS=-I/usr/include/python${PYVER} -DBUILD_SHARED_LIBS=ON ..
    make -j$(nproc)
    export LIBRARY_PATH=`pwd`:`pwd`/onnx/:$LIBRARY_PATH
    export CPLUS_INCLUDE_PATH=`pwd`:$CPLUS_INCLUDE_PATH
    export CXXFLAGS=-I`pwd`

    popd

    # Build ONNX-TensorRT
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib
    export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:/usr/local/cuda-10.2/targets/x86_64-linux/include/
    pushd .
    cd 3rdparty/onnx-tensorrt/
    mkdir -p build
    cd build
    cmake -DONNX_NAMESPACE=$ONNX_NAMESPACE ..
    make -j$(nproc)
    export LIBRARY_PATH=`pwd`:$LIBRARY_PATH
    popd

    mkdir -p /work/mxnet/lib/
    cp 3rdparty/onnx-tensorrt/third_party/onnx/build/*.so /work/mxnet/lib/
    cp -L 3rdparty/onnx-tensorrt/build/libnvonnxparser.so /work/mxnet/lib/

    cd /work/build
    cmake -DUSE_CUDA=1                            \
          -DUSE_CUDNN=1                           \
          -DUSE_OPENCV=1                          \
          -DUSE_TENSORRT=1                        \
          -DUSE_OPENMP=0                          \
          -DUSE_MKLDNN=0                          \
          -DUSE_NVML=OFF                          \
          -DUSE_MKL_IF_AVAILABLE=OFF              \
          -DMXNET_CUDA_ARCH="$CI_CMAKE_CUDA_ARCH" \
          -G Ninja                                \
          /work/mxnet

    ninja
}

build_ubuntu_gpu_mkldnn() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_CUDA=ON \
        -DUSE_NVML=OFF \
        -DMXNET_CUDA_ARCH="$CI_CMAKE_CUDA_ARCH" \
        -DBUILD_EXTENSION_PATH=/work/mxnet/example/extensions/lib_external_ops \
        -G Ninja /work/mxnet
    ninja
}

build_ubuntu_gpu_mkldnn_nocudnn() {
    set -ex

    make  \
        DEV=1                                     \
        USE_BLAS=openblas                         \
        USE_CUDA=1                                \
        USE_CUDA_PATH=/usr/local/cuda             \
        USE_CUDNN=0                               \
        USE_TVM_OP=0                              \
        CUDA_ARCH="$CI_CUDA_COMPUTE_CAPABILITIES" \
        USE_SIGNAL_HANDLER=1                      \
        -j$(nproc)
}

build_ubuntu_gpu() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_CUDA=ON \
        -DUSE_NVML=OFF \
        -DMXNET_CUDA_ARCH="$CI_CMAKE_CUDA_ARCH" \
        -DUSE_CUDNN=ON \
        -DUSE_MKLDNN=OFF \
        -DUSE_DIST_KVSTORE=ON \
        -DBUILD_CYTHON_MODULES=ON \
        -DBUILD_EXTENSION_PATH=/work/mxnet/example/extensions/lib_external_ops \
        -G Ninja /work/mxnet
    ninja
}

build_ubuntu_gpu_cuda110_cudnn8() {
    set -ex
    local CUDA_ARCH="-gencode=arch=compute_52,code=sm_52 \
        -gencode=arch=compute_70,code=sm_70 \
        -gencode=arch=compute_80,code=sm_80"
    make \
        USE_BLAS=openblas                         \
        USE_MKLDNN=0                              \
        USE_CUDA=1                                \
        USE_CUDA_PATH=/usr/local/cuda             \
        USE_CUDNN=1                               \
        USE_TVM_OP=0                              \
        USE_CPP_PACKAGE=1                         \
        USE_DIST_KVSTORE=1                        \
        CUDA_ARCH="$CUDA_ARCH"                    \
        USE_SIGNAL_HANDLER=1                      \
        -j$(nproc)
    make cython PYTHON=python3
}

build_ubuntu_gpu_cuda_cudnn_mkldnn_cpp_test() {
    set -ex
    make \
        USE_BLAS=openblas                         \
        USE_MKLDNN=1                              \
        USE_CUDA=1                                \
        USE_CUDA_PATH=/usr/local/cuda             \
        USE_CUDNN=1                               \
        USE_TVM_OP=0                              \
        USE_CPP_PACKAGE=1                         \
        USE_DIST_KVSTORE=1                        \
        CUDA_ARCH="$CI_CUDA_COMPUTE_CAPABILITIES" \
        USE_SIGNAL_HANDLER=1                      \
        PYTHON=python3                            \
        -j$(nproc)
    make test USE_CPP_PACKAGE=1 -j$(nproc)
    make cython PYTHON=python3
}

build_ubuntu_amalgamation() {
    set -ex
    # Amalgamation can not be run with -j nproc
    make -C amalgamation/ clean
    make -C amalgamation/     \
        USE_BLAS=openblas PYTHON=python3
}

build_ubuntu_amalgamation_min() {
    set -ex
    # Amalgamation can not be run with -j nproc
    make -C amalgamation/ clean
    make -C amalgamation/     \
        USE_BLAS=openblas     \
        MIN=1 PYTHON=python3
}

build_ubuntu_gpu_cmake_mkldnn() {
    set -ex
    cd /work/build
    cmake \
        -DUSE_SIGNAL_HANDLER=ON                 \
        -DUSE_CUDA=1                            \
        -DUSE_CUDNN=1                           \
        -DUSE_TVM_OP=0                          \
        -DPython3_EXECUTABLE=python3            \
        -DUSE_MKLML_MKL=1                       \
        -DCMAKE_BUILD_TYPE=Release              \
        -DMXNET_CUDA_ARCH="$CI_CMAKE_CUDA_ARCH" \
        -G Ninja                                \
        /work/mxnet

    ninja
}

build_ubuntu_gpu_cmake() {
    set -ex
    cd /work/build
    cmake \
        -DUSE_SIGNAL_HANDLER=ON                 \
        -DUSE_CUDA=ON                           \
        -DUSE_CUDNN=ON                          \
        -DUSE_TVM_OP=OFF                        \
        -DPython3_EXECUTABLE=python3            \
        -DUSE_MKL_IF_AVAILABLE=OFF              \
        -DUSE_MKLML_MKL=OFF                     \
        -DUSE_MKLDNN=OFF                        \
        -DUSE_DIST_KVSTORE=ON                   \
        -DCMAKE_BUILD_TYPE=Release              \
        -DMXNET_CUDA_ARCH="$CI_CMAKE_CUDA_ARCH" \
        -DBUILD_CYTHON_MODULES=1                \
        -G Ninja                                \
        /work/mxnet

    ninja
}

build_ubuntu_gpu_cmake_no_rtc() {
    set -ex
    cd /work/build
    cmake \
        -DUSE_SIGNAL_HANDLER=ON                 \
        -DUSE_CUDA=ON                           \
        -DUSE_CUDNN=ON                          \
        -DUSE_TVM_OP=OFF                        \
        -DPython3_EXECUTABLE=python3            \
        -DUSE_MKL_IF_AVAILABLE=OFF              \
        -DUSE_MKLML_MKL=OFF                     \
        -DUSE_MKLDNN=ON                         \
        -DUSE_DIST_KVSTORE=ON                   \
        -DCMAKE_BUILD_TYPE=Release              \
        -DMXNET_CUDA_ARCH="$CI_CMAKE_CUDA_ARCH" \
        -DBUILD_CYTHON_MODULES=1                \
        -DENABLE_CUDA_RTC=OFF                   \
        -G Ninja                                \
        /work/mxnet

    ninja
}

build_ubuntu_cpu_large_tensor() {
    set -ex
    cd /work/build
    cmake \
        -DUSE_SIGNAL_HANDLER=ON                 \
        -DUSE_CUDA=OFF                          \
        -DUSE_CUDNN=OFF                         \
        -DUSE_MKLDNN=OFF                        \
        -DCMAKE_BUILD_TYPE=Release              \
        -DUSE_INT64_TENSOR_SIZE=ON              \
        -G Ninja                                \
        /work/mxnet

    ninja
}

build_ubuntu_gpu_large_tensor() {
    set -ex
    cd /work/build
    cmake \
        -DUSE_SIGNAL_HANDLER=ON                 \
        -DUSE_CUDA=ON                           \
        -DUSE_CUDNN=ON                          \
        -DUSE_TVM_OP=OFF                        \
        -DPython3_EXECUTABLE=python3            \
        -DUSE_MKL_IF_AVAILABLE=OFF              \
        -DUSE_MKLML_MKL=OFF                     \
        -DUSE_MKLDNN=OFF                        \
        -DUSE_DIST_KVSTORE=ON                   \
        -DCMAKE_BUILD_TYPE=Release              \
        -DMXNET_CUDA_ARCH="$CI_CMAKE_CUDA_ARCH" \
        -DUSE_INT64_TENSOR_SIZE=ON              \
        -G Ninja                                \
        /work/mxnet

    ninja
}

# Testing

sanity_check() {
    set -ex
    tools/license_header.py check
    make cpplint rcpplint jnilint
    make pylint
    nosetests-3.4 tests/tutorials/test_sanity_tutorials.py
}

# Tests libmxnet
# Parameters:
# $1 -> mxnet_variant: The variant of the libmxnet.so library
# $2 -> python_cmd: The python command to use to execute the tests, python or python3
cd_unittest_ubuntu() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_MKLDNN_DEBUG=0  # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export MXNET_ENABLE_CYTHON=0
    export CD_JOB=1 # signal this is a CD run so any unecessary tests can be skipped
    export DMLC_LOG_STACK_TRACE_DEPTH=10

    local mxnet_variant=${1:?"This function requires a mxnet variant as the first argument"}
    local python_cmd=${2:?"This function requires a python command as the first argument"}

    local nose_cmd="nosetests-3.4"

    $nose_cmd $NOSE_TIMER_ARGUMENTS --verbose tests/python/unittest
    $nose_cmd $NOSE_TIMER_ARGUMENTS --verbose tests/python/quantization

    # https://github.com/apache/incubator-mxnet/issues/11801
    # if [[ ${mxnet_variant} = "cpu" ]] || [[ ${mxnet_variant} = "mkl" ]]; then
        # integrationtest_ubuntu_cpu_dist_kvstore
    # fi

    if [[ ${mxnet_variant} = cu* ]]; then
        $nose_cmd $NOSE_TIMER_ARGUMENTS --verbose tests/python/gpu

        # Adding these here as CI doesn't test all CUDA environments
        $python_cmd example/image-classification/test_score.py
        integrationtest_ubuntu_gpu_dist_kvstore
    fi

    if [[ ${mxnet_variant} = *mkl ]]; then
        # skipping python 2 testing
        # https://github.com/apache/incubator-mxnet/issues/14675
        if [[ ${python_cmd} = "python3" ]]; then
            $nose_cmd $NOSE_TIMER_ARGUMENTS --verbose tests/python/mkl
        fi
    fi
}

unittest_ubuntu_python3_cpu() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_MKLDNN_DEBUG=0  # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export MXNET_ENABLE_CYTHON=0
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    nosetests-3.4 $NOSE_COVERAGE_ARGUMENTS $NOSE_TIMER_ARGUMENTS --with-xunit --xunit-file nosetests_unittest.xml --verbose tests/python/unittest
    nosetests-3.4 $NOSE_COVERAGE_ARGUMENTS $NOSE_TIMER_ARGUMENTS --with-xunit --xunit-file nosetests_quantization.xml --verbose tests/python/quantization
}

unittest_ubuntu_python3_cpu_mkldnn() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_MKLDNN_DEBUG=0  # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export MXNET_ENABLE_CYTHON=0
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    nosetests-3.4 $NOSE_COVERAGE_ARGUMENTS $NOSE_TIMER_ARGUMENTS --with-xunit --xunit-file nosetests_unittest.xml --verbose tests/python/unittest
    nosetests-3.4 $NOSE_COVERAGE_ARGUMENTS $NOSE_TIMER_ARGUMENTS --with-xunit --xunit-file nosetests_mkl.xml --verbose tests/python/mkl
}

unittest_ubuntu_python3_gpu() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_MKLDNN_DEBUG=0 # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export CUDNN_VERSION=${CUDNN_VERSION:-7.0.3}
    export MXNET_ENABLE_CYTHON=0
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    nosetests-3.4 $NOSE_COVERAGE_ARGUMENTS $NOSE_TIMER_ARGUMENTS --with-xunit --xunit-file nosetests_gpu.xml --verbose tests/python/gpu
}

unittest_ubuntu_python3_gpu_cython() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_MKLDNN_DEBUG=1 # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export CUDNN_VERSION=${CUDNN_VERSION:-7.0.3}
    export MXNET_ENABLE_CYTHON=1
    export MXNET_ENFORCE_CYTHON=1
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    check_cython
    nosetests-3.4 $NOSE_COVERAGE_ARGUMENTS $NOSE_TIMER_ARGUMENTS --with-xunit --xunit-file nosetests_gpu.xml --verbose tests/python/gpu
}

unittest_ubuntu_python3_gpu_nocudnn() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export CUDNN_OFF_TEST_ONLY=true
    export MXNET_ENABLE_CYTHON=0
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    nosetests-3.4 $NOSE_COVERAGE_ARGUMENTS $NOSE_TIMER_ARGUMENTS --with-xunit --xunit-file nosetests_gpu.xml --verbose tests/python/gpu
}

unittest_ubuntu_tensorrt_gpu() {
    set -ex
    if [ -f /etc/redhat-release ]; then
        source /opt/rh/rh-python36/enable
    fi
    export PYTHONPATH=./python/
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export LD_LIBRARY_PATH=/work/mxnet/lib:$LD_LIBRARY_PATH
    export CUDNN_VERSION=${CUDNN_VERSION:-7.0.3}
    export MXNET_ENABLE_CYTHON=0
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100==0.24
    wget -nc http://data.mxnet.io/data/val_256_q90.rec
    python3 tests/python/tensorrt/rec2idx.py val_256_q90.rec val_256_q90.idx
    nosetests-3.4 $NOSE_COVERAGE_ARGUMENTS $NOSE_TIMER_ARGUMENTS --with-xunit --xunit-file nosetests_trt_gpu.xml --verbose --nocapture tests/python/tensorrt/
    rm val_256_q90*
}

# quantization gpu currently only runs on P3 instances
# need to separte it from unittest_ubuntu_python3_gpu()
unittest_ubuntu_python3_quantization_gpu() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_MKLDNN_DEBUG=0 # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export CUDNN_VERSION=${CUDNN_VERSION:-7.0.3}
    export MXNET_ENABLE_CYTHON=0
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    nosetests-3.4 $NOSE_COVERAGE_ARGUMENTS $NOSE_TIMER_ARGUMENTS --with-xunit --xunit-file nosetests_quantization_gpu.xml --verbose tests/python/quantization_gpu
}

unittest_ubuntu_python3_quantization_gpu_cu110() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_MKLDNN_DEBUG=0 # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export CUDNN_VERSION=${CUDNN_VERSION:-8.0.33}
    export MXNET_ENABLE_CYTHON=0
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    nosetests-3.4 $NOSE_COVERAGE_ARGUMENTS $NOSE_TIMER_ARGUMENTS --with-xunit --xunit-file nosetests_quantization_gpu.xml --verbose tests/python/quantization_gpu
}

unittest_centos7_cpu_scala() {
    set -ex
    source /opt/rh/devtoolset-7/enable
    source /opt/rh/rh-maven35/enable
    cd /work/mxnet
    scala_prepare
    cd scala-package
    mvn -B integration-test
}

unittest_ubuntu_cpu_clojure() {
    set -ex
    scala_prepare
    cd scala-package
    mvn -B install
    cd ..
    ./contrib/clojure-package/ci-test.sh
}

unittest_ubuntu_cpu_clojure_integration() {
    set -ex
    cd scala-package
    mvn -B install
    cd ..
    ./contrib/clojure-package/integration-tests.sh
}


unittest_ubuntu_cpugpu_perl() {
    set -ex
    ./perl-package/test.sh
}

unittest_cpp() {
    set -ex
    build/tests/mxnet_unit_tests
}

unittest_ubuntu_cpu_R() {
    set -ex
    mkdir -p /tmp/r-site-library
    # build R packages in parallel
    mkdir -p ~/.R/
    echo  "MAKEFLAGS = -j"$(nproc) > ~/.R/Makevars
    # make -j not supported
    make -f R-package/Makefile rpkg \
        R_LIBS=/tmp/r-site-library

    R CMD INSTALL --library=/tmp/r-site-library R-package
    make -f R-package/Makefile rpkgtest R_LIBS=/tmp/r-site-library
}

unittest_ubuntu_minimal_R() {
    set -ex
    mkdir -p /tmp/r-site-library
    # build R packages in parallel
    mkdir -p ~/.R/
    echo  "MAKEFLAGS = -j"$(nproc) > ~/.R/Makevars
    # make -j not supported
    make -f R-package/Makefile rpkg \
        R_LIBS=/tmp/r-site-library

    R CMD INSTALL --library=/tmp/r-site-library R-package
    # pick mlp as minimal R test
    R_LIBS=/tmp/r-site-library \
        Rscript -e "library(mxnet); require(mlbench); \
                    data(Sonar, package=\"mlbench\"); \
                    Sonar[,61] = as.numeric(Sonar[,61])-1; \
                    train.ind = c(1:50, 100:150); \
                    train.x = data.matrix(Sonar[train.ind, 1:60]); \
                    train.y = Sonar[train.ind, 61]; \
                    test.x = data.matrix(Sonar[-train.ind, 1:60]); \
                    test.y = Sonar[-train.ind, 61]; \
                    model = mx.mlp(train.x, train.y, hidden_node = 10, \
                                   out_node = 2, out_activation = \"softmax\", \
                                   learning.rate = 0.1, \
                                   array.layout = \"rowmajor\"); \
                    preds = predict(model, test.x, array.layout = \"rowmajor\")"
}

unittest_ubuntu_gpu_R() {
    set -ex
    mkdir -p /tmp/r-site-library
    # build R packages in parallel
    mkdir -p ~/.R/
    echo  "MAKEFLAGS = -j"$(nproc) > ~/.R/Makevars
    # make -j not supported
    make -f R-package/Makefile rpkg \
        R_LIBS=/tmp/r-site-library
    R CMD INSTALL --library=/tmp/r-site-library R-package
    make -f R-package/Makefile rpkgtest R_LIBS=/tmp/r-site-library R_GPU_ENABLE=1
}

unittest_ubuntu_cpu_julia() {
    set -ex
    export PATH="$1/bin:$PATH"
    export MXNET_HOME='/work/mxnet'
    export JULIA_DEPOT_PATH='/work/julia-depot'
    export INTEGRATION_TEST=1

    julia -e 'using InteractiveUtils; versioninfo()'

    # FIXME
    export LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libjemalloc.so'
    export LD_LIBRARY_PATH=/work/mxnet/lib:$LD_LIBRARY_PATH

    # use the prebuilt binary from $MXNET_HOME/lib
    julia --project=./julia -e 'using Pkg; Pkg.build("MXNet")'

    # run the script `julia/test/runtests.jl`
    julia --project=./julia -e 'using Pkg; Pkg.test("MXNet")'

    # See https://github.com/dmlc/MXNet.jl/pull/303#issuecomment-341171774
    julia --project=./julia -e 'using MXNet; mx._sig_checker()'
}

unittest_ubuntu_cpu_julia07() {
    set -ex
    unittest_ubuntu_cpu_julia /work/julia07
}

unittest_ubuntu_cpu_julia10() {
    set -ex
    unittest_ubuntu_cpu_julia /work/julia10
}

unittest_centos7_cpu() {
    set -ex
    source /opt/rh/rh-python36/enable
    cd /work/mxnet
    python -m "nose" $NOSE_COVERAGE_ARGUMENTS $NOSE_TIMER_ARGUMENTS --with-xunit --xunit-file nosetests_unittest.xml --verbose tests/python/unittest
    python -m "nose" $NOSE_COVERAGE_ARGUMENTS $NOSE_TIMER_ARGUMENTS --with-xunit --xunit-file nosetests_train.xml --verbose tests/python/train
}

unittest_centos7_gpu() {
    set -ex
    source /opt/rh/rh-python36/enable
    cd /work/mxnet
    export CUDNN_VERSION=${CUDNN_VERSION:-7.0.3}
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    python3 -m "nose" $NOSE_COVERAGE_ARGUMENTS $NOSE_TIMER_ARGUMENTS --with-xunit --xunit-file nosetests_gpu.xml --verbose tests/python/gpu
}

integrationtest_ubuntu_cpu_onnx() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_SUBGRAPH_VERBOSE=0
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    COV_ARG="--cov=./ --cov-report=xml --cov-append"
    pytest $COV_ARG --verbose tests/python-pytest/onnx/test_operators.py
    pytest $COV_ARG --verbose tests/python-pytest/onnx/mxnet_export_test.py
    # Skip this as https://github.com/apache/incubator-mxnet/pull/19914 breaks import
    #pytest $COV_ARG --verbose tests/python-pytest/onnx/test_models.py
    #pytest $COV_ARG --verbose tests/python-pytest/onnx/test_node.py
    pytest $COV_ARG --verbose tests/python-pytest/onnx/test_onnxruntime.py::test_bert_inference_onnxruntime
    pytest $COV_ARG --verbose tests/python-pytest/onnx/test_onnxruntime.py::test_obj_class_model_inference_onnxruntime[mobilenetv3_large]
    pytest $COV_ARG --verbose tests/python-pytest/onnx/test_onnxruntime.py::test_obj_class_model_inference_onnxruntime[resnest200]
    pytest $COV_ARG --verbose tests/python-pytest/onnx/test_onnxruntime.py::test_obj_class_model_inference_onnxruntime[resnet50_v2]
    pytest $COV_ARG --verbose tests/python-pytest/onnx/test_onnxruntime.py::test_obj_class_model_inference_onnxruntime[vgg19_bn]
    pytest $COV_ARG --verbose tests/python-pytest/onnx/test_onnxruntime.py::test_obj_detection_model_inference_onnxruntime[center_net_resnet101_v1b_voc]
    pytest $COV_ARG --verbose tests/python-pytest/onnx/test_onnxruntime.py::test_img_segmentation_model_inference_onnxruntime[deeplab_resnet50_citys]
    pytest $COV_ARG --verbose tests/python-pytest/onnx/test_onnxruntime.py::test_pose_estimation_model_inference_onnxruntime[mobile_pose_mobilenet1.0]
    pytest $COV_ARG --verbose tests/python-pytest/onnx/test_onnxruntime.py::test_action_recognition_model_inference_onnxruntime[inceptionv3_kinetics400]
}

integrationtest_ubuntu_gpu_python() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    example/image-classification/test_score.py
}

integrationtest_ubuntu_gpu_caffe() {
    set -ex
    export PYTHONPATH=/work/deps/caffe/python:./python
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    tools/caffe_converter/test_converter.py
}

integrationtest_ubuntu_cpu_asan() {
    set -ex
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.5
    export DMLC_LOG_STACK_TRACE_DEPTH=10

    cd /work/mxnet/build/cpp-package/example/
    /work/mxnet/cpp-package/example/get_data.sh
    ./mlp_cpu
}

integrationtest_ubuntu_gpu_cpp_package() {
    set -ex
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    cpp-package/tests/ci_test.sh
}

integrationtest_ubuntu_gpu_capi_cpp_package() {
    set -ex
    export PYTHONPATH=./python/
    export LD_LIBRARY_PATH=/work/mxnet/lib:$LD_LIBRARY_PATH
    python3 -c "import mxnet as mx; mx.test_utils.download_model(\"imagenet1k-resnet-18\"); mx.test_utils.download_model(\"imagenet1k-resnet-152\"); mx.test_utils.download_model(\"imagenet1k-resnet-50\");"
    # Load symbol, convert symbol to leverage fusion with subgraphs, save the model
    python3 -c "import mxnet as mx; x = mx.sym.load(\"imagenet1k-resnet-152-symbol.json\"); x.get_backend_symbol(\"MKLDNN\"); x.save(\"imagenet1k-resnet-152-subgraph-symbol.json\");"
    # Copy params file with a different name, used in subgraph symbol testing
    cp imagenet1k-resnet-152-0000.params imagenet1k-resnet-152-subgraph-0000.params
    build/tests/cpp/mxnet_unit_tests --gtest_filter="ThreadSafety.*"
    build/tests/cpp/mxnet_unit_tests --gtest_filter="ThreadSafety.*" --thread-safety-with-cpu
    # Also run thread safety tests in NaiveEngine mode
    export MXNET_ENGINE_TYPE=NaiveEngine
    build/tests/cpp/mxnet_unit_tests --gtest_filter="ThreadSafety.*"
    build/tests/cpp/mxnet_unit_tests --gtest_filter="ThreadSafety.*" --thread-safety-with-cpu
    unset MXNET_ENGINE_TYPE
}

integrationtest_ubuntu_cpu_dist_kvstore() {
    set -ex
    pushd .
    export PYTHONPATH=./python/
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export MXNET_USE_OPERATOR_TUNING=0
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    cd tests/nightly/
    python3 ../../tools/launch.py -n 7 --launcher local python3 dist_sync_kvstore.py --type=gluon_step_cpu
    python3 ../../tools/launch.py -n 7 --launcher local python3 dist_sync_kvstore.py --type=gluon_sparse_step_cpu
    python3 ../../tools/launch.py -n 7 --launcher local python3 dist_sync_kvstore.py --type=invalid_cpu
    python3 ../../tools/launch.py -n 7 --launcher local python3 dist_sync_kvstore.py --type=gluon_type_cpu
    python3 ../../tools/launch.py -n 7 --launcher local python3 dist_sync_kvstore.py
    python3 ../../tools/launch.py -n 7 --launcher local python3 dist_sync_kvstore.py --no-multiprecision
    python3 ../../tools/launch.py -n 7 --launcher local python3 dist_sync_kvstore.py --type=compressed_cpu
    python3 ../../tools/launch.py -n 7 --launcher local python3 dist_sync_kvstore.py --type=compressed_cpu --no-multiprecision
    python3 ../../tools/launch.py -n 3 --launcher local python3 test_server_profiling.py
    popd
}

integrationtest_ubuntu_cpu_scala() {
    set -ex
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    scala_prepare
    cd scala-package
    mvn -B verify -DskipTests=false
}

integrationtest_ubuntu_gpu_scala() {
    set -ex
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    scala_prepare
    cd scala-package
    export SCALA_TEST_ON_GPU=1
    mvn -B verify -DskipTests=false
}

integrationtest_ubuntu_gpu_dist_kvstore() {
    set -ex
    pushd .
    export PYTHONPATH=./python/
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    cd tests/nightly/
    python3 ../../tools/launch.py -n 4 --launcher local python3 dist_device_sync_kvstore.py
    python3 ../../tools/launch.py -n 4 --launcher local python3 dist_device_sync_kvstore_custom.py
    python3 ../../tools/launch.py --p3 -n 4 --launcher local python3 dist_device_sync_kvstore_custom.py
    python3 ../../tools/launch.py -n 4 --launcher local python3 dist_sync_kvstore.py --type=init_gpu
    popd
}

test_ubuntu_cpu_python3() {
    set -ex
    pushd .
    export MXNET_LIBRARY_PATH=/work/build/libmxnet.so
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    VENV=mxnet_py3_venv
    virtualenv -p `which python3` $VENV
    source $VENV/bin/activate

    cd /work/mxnet/python
    pip3 install nose nose-timer
    pip3 install -e .
    cd /work/mxnet
    python3 -m "nose" $NOSE_COVERAGE_ARGUMENTS $NOSE_TIMER_ARGUMENTS --verbose tests/python/unittest

    popd
}

# QEMU based ARM tests
unittest_ubuntu_python3_arm() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_MKLDNN_DEBUG=0  # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export MXNET_ENABLE_CYTHON=0
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    python3 -m nose --verbose tests/python/unittest/test_engine.py
}

# Functions that run the nightly Tests:

#Runs Apache RAT Check on MXNet Source for License Headers
test_rat_check() {
    set -e
    set -o pipefail
    pushd .

    cd /usr/local/src/apache-rat-0.13

    # Use shell number 5 to duplicate the log output. It get sprinted and stored in $OUTPUT at the same time https://stackoverflow.com/a/12451419
    exec 5>&1
    OUTPUT=$(java -jar apache-rat-0.13.jar -E /work/mxnet/rat-excludes -d /work/mxnet|tee >(cat - >&5))
    ERROR_MESSAGE="Printing headers for text files without a valid license header"


    echo "-------Process The Output-------"

    if [[ $OUTPUT =~ $ERROR_MESSAGE ]]; then
        echo "ERROR: RAT Check detected files with unknown licenses. Please fix and run test again!";
        exit 1
    else
        echo "SUCCESS: There are no files with an Unknown License.";
    fi
    popd
}

#Checks MXNet for Compilation Warnings
nightly_test_compilation_warning() {
    set -ex
    export PYTHONPATH=./python/
    ./tests/nightly/compilation_warnings/compilation_warnings.sh
}

#Checks the MXNet Installation Guide - currently checks pip, build from source and virtual env on cpu and gpu
nightly_test_installation() {
    set -ex
    # The run_test_installation_docs.sh expects the path to index.md and the first and last line numbers of the index.md file
    # First execute the test script and then call the method specified by the Jenkinsfile - ${1}
    source ./tests/jenkins/run_test_installation_docs.sh docs/install/index.md 1 1686; ${1}
}

# Runs Imagenet inference
nightly_test_imagenet_inference() {
    set -ex
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    echo $PWD
    cp /work/mxnet/build/cpp-package/example/imagenet_inference /work/mxnet/cpp-package/example/inference/
    cd /work/mxnet/cpp-package/example/inference/
    ./unit_test_imagenet_inference.sh
}

#Runs a simple MNIST training example
nightly_test_image_classification() {
    set -ex
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    ./tests/nightly/test_image_classification.sh
}

#Single Node KVStore Test
nightly_test_KVStore_singleNode() {
    set -ex
    export PYTHONPATH=./python/
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    tests/nightly/test_kvstore.py
}

#Test Large Tensor Size
nightly_test_large_tensor() {
    set -ex
    export PYTHONPATH=./python/
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    nosetests-3.4 tests/nightly/test_large_array.py:test_tensor
    nosetests-3.4 tests/nightly/test_large_array.py:test_nn
    nosetests-3.4 tests/nightly/test_large_array.py:test_basic
}

#Test Large Vectors
nightly_test_large_vector() {
    set -ex
    export PYTHONPATH=./python/
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    nosetests-3.4 tests/nightly/test_large_vector.py:test_tensor
    nosetests-3.4 tests/nightly/test_large_vector.py:test_nn
    nosetests-3.4 tests/nightly/test_large_vector.py:test_basic
}

#Test Large Vectors
nightly_test_large_vector() {
    set -ex
    export PYTHONPATH=./python/
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    nosetests-3.4 tests/nightly/test_large_vector.py:test_tensor
    nosetests-3.4 tests/nightly/test_large_vector.py:test_nn
    nosetests-3.4 tests/nightly/test_large_vector.py:test_basic
}

#Tests Amalgamation Build with 5 different sets of flags
nightly_test_amalgamation() {
    set -ex
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    # Amalgamation can not be run with -j nproc
    make -C amalgamation/ clean
    make -C amalgamation/ ${1} ${2}
}

#Tests Amalgamation Build for Javascript
nightly_test_javascript() {
    set -ex
    export LLVM=/work/deps/emscripten-fastcomp/build/bin
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    # This part is needed to run emcc correctly
    cd /work/deps/emscripten
    ./emcc
    touch ~/.emscripten
    make -C /work/mxnet/amalgamation libmxnet_predict.js MIN=1 EMCC=/work/deps/emscripten/emcc
}

#Tests Model backwards compatibility on MXNet
nightly_model_backwards_compat_test() {
    set -ex
    export PYTHONPATH=/work/mxnet/python/
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    ./tests/nightly/model_backwards_compatibility_check/model_backward_compat_checker.sh
}

#Backfills S3 bucket with models trained on earlier versions of mxnet
nightly_model_backwards_compat_train() {
    set -ex
    export PYTHONPATH=./python/
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    ./tests/nightly/model_backwards_compatibility_check/train_mxnet_legacy_models.sh
}

nightly_straight_dope_python3_single_gpu_tests() {
    set -ex
    cd /work/mxnet/tests/nightly/straight_dope
    export PYTHONPATH=/work/mxnet/python/
    export MXNET_TEST_KERNEL=python3
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    nosetests-3.4 $NOSE_TIMER_ARGUMENTS --with-xunit --xunit-file nosetests_straight_dope_python3_single_gpu.xml \
      test_notebooks_single_gpu.py --nologcapture
}

nightly_straight_dope_python3_multi_gpu_tests() {
    set -ex
    cd /work/mxnet/tests/nightly/straight_dope
    export PYTHONPATH=/work/mxnet/python/
    export MXNET_TEST_KERNEL=python3
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    nosetests-3.4 $NOSE_TIMER_ARGUMENTS --with-xunit --xunit-file nosetests_straight_dope_python3_multi_gpu.xml \
      test_notebooks_multi_gpu.py --nologcapture
}

nightly_tutorial_test_ubuntu_python3_gpu() {
    set -ex
    cd /work/mxnet/docs
    export BUILD_VER=tutorial
    export MXNET_DOCS_BUILD_MXNET=0
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    make html
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export PYTHONPATH=/work/mxnet/python/
    export MXNET_TUTORIAL_TEST_KERNEL=python3
    cd /work/mxnet/tests/tutorials
    nosetests-3.4 $NOSE_TIMER_ARGUMENTS --with-xunit --xunit-file nosetests_tutorials.xml test_tutorials.py --nologcapture
}

nightly_java_demo_test_cpu() {
    set -ex
    cd /work/mxnet/scala-package/mxnet-demo/java-demo
    mvn -B -Pci-nightly install
    bash bin/java_sample.sh
    bash bin/run_od.sh
}

nightly_scala_demo_test_cpu() {
    set -ex
    cd /work/mxnet/scala-package/mxnet-demo/scala-demo
    mvn -B -Pci-nightly install
    bash bin/demo.sh
    bash bin/run_im.sh
}

nightly_estimator() {
    set -ex
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    cd /work/mxnet/tests/nightly/estimator
    export PYTHONPATH=/work/mxnet/python/
    nosetests test_estimator_cnn.py
    nosetests test_sentiment_rnn.py
}

nightly_onnx_tests() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_SUBGRAPH_VERBOSE=0
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    COV_ARG="--cov=./ --cov-report=xml --cov-append"
    pytest $COV_ARG --verbose tests/python-pytest/onnx/test_onnxruntime.py
}

# For testing PRs
deploy_docs() {
    set -ex
    pushd .

    # Setup for Julia docs
    export PATH="/work/julia10/bin:$PATH"
    export MXNET_HOME='/work/mxnet'
    export JULIA_DEPOT_PATH='/work/julia-depot'

    julia -e 'using InteractiveUtils; versioninfo()'

    # FIXME
    export LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libjemalloc.so'
    export LD_LIBRARY_PATH=/work/mxnet/lib:$LD_LIBRARY_PATH
    # End Julia setup

    build_python_docs

    popd
}


build_docs_setup() {
    build_folder="docs/_build"
    mxnetlib_folder="/work/mxnet/lib"

    mkdir -p $build_folder
    mkdir -p $mxnetlib_folder
}

build_ubuntu_cpu_docs() {
    set -ex
    export CC="gcc"
    export CXX="g++"
    make \
        DEV=1                         \
        USE_CPP_PACKAGE=1             \
        USE_BLAS=openblas             \
        USE_MKLDNN=0                  \
        USE_DIST_KVSTORE=1            \
        USE_LIBJPEG_TURBO=1           \
        USE_SIGNAL_HANDLER=1          \
        -j$(nproc)
}


build_jekyll_docs() {
    set -ex
    source /etc/profile.d/rvm.sh

    pushd .
    build_docs_setup
    pushd docs/static_site
    make clean
    make html
    popd

    GZIP=-9 tar zcvf jekyll-artifacts.tgz -C docs/static_site/build html
    mv jekyll-artifacts.tgz docs/_build/
    popd
}


build_python_docs() {
   set -ex
   pushd .

   build_docs_setup

   pushd docs/python_docs
   eval "$(/work/miniconda/bin/conda shell.bash hook)"
   conda env create -f environment.yml -p /work/conda_env
   conda activate /work/conda_env
   pip install themes/mx-theme
   pip install -e /work/mxnet/python --user

   pushd python
   make clean
   make html EVAL=0

   GZIP=-9 tar zcvf python-artifacts.tgz -C build/_build/html .
   popd

   mv python/python-artifacts.tgz /work/mxnet/docs/_build/
   popd

   popd
}


build_c_docs() {
    set -ex
    pushd .

    build_docs_setup
    doc_path="docs/cpp_docs"
    pushd $doc_path

    make clean
    make html

    doc_artifact="c-artifacts.tgz"
    GZIP=-9 tar zcvf $doc_artifact -C build/html/html .
    popd

    mv $doc_path/$doc_artifact docs/_build/

    popd
}


build_r_docs() {
    set -ex
    pushd .

    build_docs_setup
    r_root='R-package'
    r_pdf='mxnet-r-reference-manual.pdf'
    r_build='build'
    docs_build_path="$r_root/$r_build/$r_pdf"
    artifacts_path='docs/_build/r-artifacts.tgz'

    mkdir -p $r_root/$r_build

    unittest_ubuntu_minimal_R

    pushd $r_root

    R_LIBS=/tmp/r-site-library R CMD Rd2pdf . --no-preview --encoding=utf8 -o $r_build/$r_pdf

    popd

    GZIP=-9 tar zcvf $artifacts_path $docs_build_path

    popd
}


build_scala() {
   set -ex
   pushd .

   cd scala-package
   mvn -B install -DskipTests

   popd
}


build_scala_docs() {
    set -ex
    pushd .
    build_docs_setup
    build_scala

    scala_path='scala-package'
    docs_build_path='scala-package/docs/build/docs/scala'
    artifacts_path='docs/_build/scala-artifacts.tgz'

    pushd $scala_path

    scala_doc_sources=`find . -type f -name "*.scala" | egrep "./core|./infer" | egrep -v "/javaapi"  | egrep -v "Suite" | egrep -v "/mxnetexamples"`
    jar_native=`find native -name "*.jar" | grep "target/lib/" | tr "\\n" ":" `
    jar_macros=`find macros -name "*.jar" | tr "\\n" ":" `
    jar_core=`find core -name "*.jar" | tr "\\n" ":" `
    jar_infer=`find infer -name "*.jar" | tr "\\n" ":" `
    scala_doc_classpath=$jar_native:$jar_macros:$jar_core:$jar_infer

    scala_ignore_errors=''
    legacy_ver=".*1.2|1.3.*"
    # BUILD_VER needs to be pull from environment vars
    if [[ $_BUILD_VER =~ $legacy_ver ]]
    then
      # There are unresolvable errors on mxnet 1.2.x. We are ignoring those
      # errors while aborting the ci on newer versions
      echo "We will ignoring unresolvable errors on MXNet 1.2/1.3."
      scala_ignore_errors='; exit 0'
    fi

    scaladoc $scala_doc_sources -classpath $scala_doc_classpath $scala_ignore_errors -doc-title MXNet
    popd

    # Clean-up old artifacts
    rm -rf $docs_build_path
    mkdir -p $docs_build_path

    for doc_file in index index.html org lib index.js package.html; do
        mv $scala_path/$doc_file $docs_build_path
    done

    GZIP=-9 tar -zcvf $artifacts_path -C $docs_build_path .

    popd
}


build_julia_docs() {
   set -ex
   pushd .

   build_docs_setup
   # Setup environment for Julia docs
   export PATH="/work/julia10/bin:$PATH"
   export MXNET_HOME='/work/mxnet'
   export JULIA_DEPOT_PATH='/work/julia-depot'
   export LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libjemalloc.so'
   export LD_LIBRARY_PATH=/work/mxnet/lib:$LD_LIBRARY_PATH

   julia_doc_path='julia/docs/site/'
   julia_doc_artifact='docs/_build/julia-artifacts.tgz'

   echo "Julia will check for MXNet in $MXNET_HOME/lib"


   make -C julia/docs

   GZIP=-9 tar -zcvf $julia_doc_artifact -C $julia_doc_path .

   popd
}


build_java_docs() {
    set -ex
    pushd .

    build_docs_setup
    build_scala

    # Re-use scala-package build artifacts.
    java_path='scala-package'
    docs_build_path='docs/scala-package/build/docs/java'
    artifacts_path='docs/_build/java-artifacts.tgz'

    pushd $java_path

    java_doc_sources=`find . -type f -name "*.scala" | egrep "./core|./infer"  | egrep "/javaapi"  | egrep -v "Suite" | egrep -v "/mxnetexamples"`
    jar_native=`find native -name "*.jar" | grep "target/lib/" | tr "\\n" ":" `
    jar_macros=`find macros -name "*.jar" | tr "\\n" ":" `
    jar_core=`find core -name "*.jar" | tr "\\n" ":" `
    jar_infer=`find infer -name "*.jar" | tr "\\n" ":" `
    java_doc_classpath=$jar_native:$jar_macros:$jar_core:$jar_infer

    scaladoc $java_doc_sources -classpath $java_doc_classpath -feature -deprecation -doc-title MXNet
    popd

    # Clean-up old artifacts
    rm -rf $docs_build_path
    mkdir -p $docs_build_path

    for doc_file in index index.html org lib index.js package.html; do
        mv $java_path/$doc_file $docs_build_path
    done

    GZIP=-9 tar -zcvf $artifacts_path -C $docs_build_path .

    popd
}


build_clojure_docs() {
    set -ex
    pushd .

    build_docs_setup
    build_scala

    clojure_path='contrib/clojure-package'
    clojure_doc_path='contrib/clojure-package/target/doc'
    clojure_doc_artifact='docs/_build/clojure-artifacts.tgz'

    pushd $clojure_path
    lein codox
    popd

    GZIP=-9 tar -zcvf $clojure_doc_artifact -C $clojure_doc_path .

    popd
}

build_docs() {
    pushd docs/_build
    tar -xzf jekyll-artifacts.tgz
    api_folder='html/api'
    # Python has it's own landing page/site so we don't put it in /docs/api
    mkdir -p $api_folder/python/docs && tar -xzf python-artifacts.tgz --directory $api_folder/python/docs
    mkdir -p $api_folder/cpp/docs/api && tar -xzf c-artifacts.tgz --directory $api_folder/cpp/docs/api
    mkdir -p $api_folder/r/docs/api && tar -xzf r-artifacts.tgz --directory $api_folder/r/docs/api
    mkdir -p $api_folder/julia/docs/api && tar -xzf julia-artifacts.tgz --directory $api_folder/julia/docs/api
    mkdir -p $api_folder/scala/docs/api && tar -xzf scala-artifacts.tgz --directory $api_folder/scala/docs/api
    mkdir -p $api_folder/java/docs/api && tar -xzf java-artifacts.tgz --directory $api_folder/java/docs/api
    mkdir -p $api_folder/clojure/docs/api && tar -xzf clojure-artifacts.tgz --directory $api_folder/clojure/docs/api
    GZIP=-9 tar -zcvf full_website.tgz -C html .
    popd
}

build_docs_beta() {
    pushd docs/_build
    tar -xzf jekyll-artifacts.tgz
    python_doc_folder="html/versions/$BRANCH/api/python/docs"
    mkdir -p $python_doc_folder && tar -xzf python-artifacts.tgz --directory $python_doc_folder
    GZIP=-9 tar -zcvf beta_website.tgz -C html .
    popd
}

push_docs() {
    folder_name=$1
    set -ex
    pip3 install --user awscli
    export PATH=~/.local/bin:$PATH
    pushd docs/_build
    wget https://mxnet-website-static-artifacts.s3.us-east-2.amazonaws.com/versions.zip && unzip versions.zip && rm versions.zip
    mkdir $folder_name && tar -xzf full_website.tgz -C $folder_name --strip-components 1
    # check if folder_name already exists in versions
    pushd versions
    if [ -d "$folder_name" ]; then
        echo "Folder $folder_name already exists in versions. Please double check the FOLDER_NAME variable in Jenkens pipeline"
        exit 1
    fi
    popd
    mv $folder_name versions
    zip -r9 versions.zip versions/.
    aws s3 cp versions.zip s3://mxnet-website-static-artifacts --acl public-read
    popd
}

create_repo() {
   repo_folder=$1
   mxnet_url=$2
   git clone $mxnet_url $repo_folder --recursive
   echo "Adding MXNet upstream repo..."
   cd $repo_folder
   git remote add upstream https://github.com/apache/incubator-mxnet
   cd ..
}


refresh_branches() {
   repo_folder=$1
   cd $repo_folder
   git fetch
   git fetch upstream
   cd ..
}

checkout() {
   repo_folder=$1
   cd $repo_folder
   # Overriding configs later will cause a conflict here, so stashing...
   git stash
   # Fails to checkout if not available locally, so try upstream
   git checkout "$repo_folder" || git branch $repo_folder "upstream/$repo_folder" && git checkout "$repo_folder" || exit 1
   if [ $tag == 'master' ]; then
      git pull
      # master gets warnings as errors for Sphinx builds
      OPTS="-W"
      else
      OPTS=
   fi
   git submodule update --init --recursive
   cd ..
}

build_static_libmxnet() {
    set -ex
    pushd .
    source /opt/rh/devtoolset-8/enable
    source /opt/rh/rh-python36/enable
    # Opt in to newer GCC C++ ABI. devtoolset defaults to ABI Version 2.
    export CXXFLAGS="-fabi-version=11 -fabi-compat-version=7"
    local mxnet_variant=${1:?"This function requires a python command as the first argument"}
    source tools/staticbuild/build.sh ${mxnet_variant}
    popd
}

# Tests CD PyPI packaging in CI
ci_package_pypi() {
    set -ex
    # copies mkldnn header files to 3rdparty/mkldnn/include/oneapi/dnnl/ as in CD
    mkdir -p 3rdparty/mkldnn/include/oneapi/dnnl
    cp include/mkldnn/oneapi/dnnl/dnnl_version.h 3rdparty/mkldnn/include/oneapi/dnnl/.
    cp include/mkldnn/oneapi/dnnl/dnnl_config.h 3rdparty/mkldnn/include/oneapi/dnnl/.
    local mxnet_variant=${1:?"This function requires a python command as the first argument"}
    cd_package_pypi ${mxnet_variant}
    cd_integration_test_pypi
}

# Packages libmxnet into wheel file
cd_package_pypi() {
    set -ex
    pushd .
    source /opt/rh/devtoolset-8/enable
    source /opt/rh/rh-python36/enable
    # Opt in to newer GCC C++ ABI. devtoolset defaults to ABI Version 2.
    export CXXFLAGS="-fabi-version=11 -fabi-compat-version=7"
    local mxnet_variant=${1:?"This function requires a python command as the first argument"}
    ./cd/python/pypi/pypi_package.sh ${mxnet_variant}
    popd
}

# Sanity checks wheel file
cd_integration_test_pypi() {
    set -ex
    source /opt/rh/rh-python36/enable

    # install mxnet wheel package
    pip3 install --user ./wheel_build/dist/*.whl

    # execute tests
    # TODO: Add tests (18549)
}

# Publishes wheel to PyPI
cd_pypi_publish() {
    set -ex
    pip3 install --user twine
    python3 ./cd/python/pypi/pypi_publish.py `readlink -f wheel_build/dist/*.whl`
}

cd_s3_publish() {
    set -ex
    pip3 install --upgrade --user awscli
    filepath=$(readlink -f wheel_build/dist/*.whl)
    filename=$(basename $filepath)
    variant=$(echo $filename | cut -d'-' -f1 | cut -d'_' -f2 -s)
    if [ -z "${variant}" ]; then
        variant="cpu"
    fi
    export PATH=/usr/local/bin:$PATH
    aws s3 cp ${filepath} s3://apache-mxnet/dist/python/${variant}/${filename} --grants read=uri=http://acs.amazonaws.com/groups/global/AllUsers full=id=43f628fab72838a4f0b929d7f1993b14411f4b0294b011261bc6bd3e950a6822
}

build_static_scala_cpu() {
    set -ex
    pushd .
    scala_prepare
    export MAVEN_PUBLISH_OS_TYPE=linux-x86_64-cpu
    export mxnet_variant=cpu
    source /opt/rh/rh-maven35/enable
    ./ci/publish/scala/build.sh
    popd
}

build_static_python_cpu() {
    set -ex
    pushd .
    export mxnet_variant=cpu
    source /opt/rh/devtoolset-8/enable
    source /opt/rh/rh-python36/enable
    # Opt in to newer GCC C++ ABI. devtoolset defaults to ABI Version 2.
    export CXXFLAGS="-fabi-version=11 -fabi-compat-version=7"
    ./ci/publish/python/build.sh
    popd
}

build_static_python_cu101() {
    set -ex
    pushd .
    export mxnet_variant=cu101
    source /opt/rh/devtoolset-8/enable
    source /opt/rh/rh-python36/enable
    # Opt in to newer GCC C++ ABI. devtoolset defaults to ABI Version 2.
    export CXXFLAGS="-fabi-version=11 -fabi-compat-version=7"
    ./ci/publish/python/build.sh
    popd
}

build_static_python_cu102() {
    set -ex
    pushd .
    export mxnet_variant=cu102
    source /opt/rh/devtoolset-8/enable
    source /opt/rh/rh-python36/enable
    # Opt in to newer GCC C++ ABI. devtoolset defaults to ABI Version 2.
    export CXXFLAGS="-fabi-version=11 -fabi-compat-version=7"
    ./ci/publish/python/build.sh
    popd
}

build_static_python_cpu_cmake() {
    set -ex
    pushd .
    export mxnet_variant=cpu
    export CMAKE_STATICBUILD=1
    ./ci/publish/python/build.sh
    popd
}

build_static_python_cu101_cmake() {
    set -ex
    pushd .
    export mxnet_variant=cu101
    export CMAKE_STATICBUILD=1
    ./ci/publish/python/build.sh
    popd
}

publish_scala_build() {
    set -ex
    pushd .
    scala_prepare
    ./ci/publish/scala/build.sh
    popd
}

publish_scala_test() {
    set -ex
    pushd .
    scala_prepare
    ./ci/publish/scala/test.sh
    popd
}

publish_scala_deploy() {
    set -ex
    pushd .
    scala_prepare
    ./ci/publish/scala/deploy.sh
    popd
}

# artifact repository unit tests
test_artifact_repository() {
    set -ex
    pushd .
    cd cd/utils/
    OMP_NUM_THREADS=$(expr $(nproc) / 4) pytest -n 4 test_artifact_repository.py
    popd
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
