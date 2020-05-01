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

build_ccache_wrappers() {
    set -ex

    if [ -z ${CC+x} ]; then
        echo "No \$CC set, defaulting to gcc";
        export CC=gcc
    fi
     if [ -z ${CXX+x} ]; then
       echo "No \$CXX set, defaulting to g++";
       export CXX=g++
    fi

    # Recommended by CCache: https://ccache.samba.org/manual.html#_run_modes
    # Add to the beginning of path to ensure this redirection is picked up instead
    # of the original ones. Especially CUDA/NVCC appends itself to the beginning of the
    # path and thus this redirect is ignored. This change fixes this problem
    # This hacky approach with symbolic links is required because underlying build
    # systems of our submodules ignore our CMake settings. If they use Makefile,
    # we can't influence them at all in general and NVCC also prefers to hardcode their
    # compiler instead of respecting the settings. Thus, we take this brutal approach
    # and just redirect everything of this installer has been called.
    # In future, we could do these links during image build time of the container.
    # But in the beginning, we'll make this opt-in. In future, loads of processes like
    # the scala make step or numpy compilation and other pip package generations
    # could be heavily sped up by using ccache as well.
    mkdir -p /tmp/ccache-redirects
    export PATH=/tmp/ccache-redirects:$PATH
    CCACHE=`which ccache`
    ln -sf $CCACHE /tmp/ccache-redirects/gcc
    ln -sf $CCACHE /tmp/ccache-redirects/gcc-8
    ln -sf $CCACHE /tmp/ccache-redirects/g++
    ln -sf $CCACHE /tmp/ccache-redirects/g++-8
    ln -sf $CCACHE /tmp/ccache-redirects/clang++-3.9
    ln -sf $CCACHE /tmp/ccache-redirects/clang-3.9
    ln -sf $CCACHE /tmp/ccache-redirects/clang++-5.0
    ln -sf $CCACHE /tmp/ccache-redirects/clang-5.0
    ln -sf $CCACHE /tmp/ccache-redirects/clang++-6.0
    ln -sf $CCACHE /tmp/ccache-redirects/clang-6.0
    ln -sf $CCACHE /tmp/ccache-redirects/clang++-10
    ln -sf $CCACHE /tmp/ccache-redirects/clang-10
    #Doesn't work: https://github.com/ccache/ccache/issues/373
    # ln -sf $CCACHE /tmp/ccache-redirects/nvcc
    # ln -sf $CCACHE /tmp/ccache-redirects/nvcc
    # export NVCC="/tmp/ccache-redirects/nvcc"

    # Uncomment if you would like to debug CCache hit rates.
    # You can monitor using tail -f ccache-log
    #export CCACHE_LOGFILE=/work/mxnet/ccache-log
    #export CCACHE_LOGFILE=/tmp/ccache-log
    #export CCACHE_DEBUG=1
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

# Compiles the dynamic mxnet library
# Parameters:
# $1 -> mxnet_variant: the mxnet variant to build, e.g. cpu, native, cu100, cu92, etc.
build_dynamic_libmxnet() {
    set -ex

    local mxnet_variant=${1:?"This function requires a mxnet variant as the first argument"}

    # relevant licenses will be placed in the licenses directory
    gather_licenses

    cd /work/build
    source /opt/rh/devtoolset-7/enable
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
        -DENABLE_CUDA_RTC=OFF \
        -DSUPPORT_F16C=OFF \
        -DUSE_OPENCV=OFF \
        -DUSE_OPENMP=ON \
        -DUSE_LAPACK=OFF \
        -DUSE_SIGNAL_HANDLER=ON \
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

    # Lapack functionality will be included and statically linked to openblas.
    # But USE_LAPACK needs to be set to OFF, otherwise the main CMakeLists.txt
    # file tries to add -llapack. Lapack functionality though, requires -lgfortran
    # to be linked additionally.

    # We do not need OpenMP, since most armv6 systems have only 1 core

    cmake \
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
        -DUSE_CUDA=OFF \
        -DUSE_OPENCV=OFF \
        -DUSE_OPENMP=OFF \
        -DUSE_SIGNAL_HANDLER=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_LAPACK=OFF \
        -DBUILD_CPP_EXAMPLES=OFF \
        -Dmxnet_LINKER_LIBS=-latomic \
        -G Ninja /work/mxnet

    ninja
    build_wheel
}

build_armv7() {
    set -ex
    cd /work/build

    # Lapack functionality will be included and statically linked to openblas.
    # But USE_LAPACK needs to be set to OFF, otherwise the main CMakeLists.txt
    # file tries to add -llapack. Lapack functionality though, requires -lgfortran
    # to be linked additionally.

    cmake \
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
        -DCMAKE_CROSSCOMPILING=ON \
        -DUSE_CUDA=OFF \
        -DUSE_OPENCV=OFF \
        -DUSE_OPENMP=ON \
        -DUSE_SIGNAL_HANDLER=ON \
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
        -DSUPPORT_F16C=OFF \
        -DUSE_OPENCV=OFF \
        -DUSE_OPENMP=ON \
        -DUSE_LAPACK=OFF \
        -DUSE_SIGNAL_HANDLER=ON \
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
    cmake \
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
        -DANDROID_ABI="armeabi-v7a" \
        -DANDROID_STL="c++_shared" \
        -DANDROID=ON \
        -DUSE_CUDA=OFF \
        -DUSE_SSE=OFF \
        -DSUPPORT_F16C=OFF \
        -DUSE_LAPACK=OFF \
        -DUSE_OPENCV=OFF \
        -DUSE_OPENMP=OFF \
        -DUSE_SIGNAL_HANDLER=ON \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -G Ninja /work/mxnet
    ninja
}

build_android_armv8() {
    set -ex
    cd /work/build
    cmake \
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
        -DANDROID_ABI="arm64-v8a" \
        -DANDROID_STL="c++_shared" \
        -DANDROID=ON \
        -DUSE_CUDA=OFF \
        -DUSE_SSE=OFF \
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
    cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DENABLE_TESTCOVERAGE=ON \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_MKLDNN=OFF \
        -DUSE_DIST_KVSTORE=ON \
        -DUSE_CUDA=OFF \
        -G Ninja /work/mxnet
    ninja
}

build_centos7_cpu_make() {
    set -ex
    cd /work/mxnet
    source /opt/rh/devtoolset-7/enable
    make \
        DEV=1 \
        USE_LAPACK=1 \
        USE_LAPACK_PATH=/usr/lib64/liblapack.so \
        USE_BLAS=openblas \
        USE_MKLDNN=0 \
        USE_DIST_KVSTORE=1 \
        USE_SIGNAL_HANDLER=1 \
        -j$(nproc)
}

build_centos7_mkldnn() {
    set -ex
    cd /work/build
    source /opt/rh/devtoolset-7/enable
    cmake \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_MKLDNN=ON \
        -DUSE_CUDA=OFF \
        -G Ninja /work/mxnet
    ninja
}

build_centos7_gpu() {
    set -ex
    cd /work/build
    source /opt/rh/devtoolset-7/enable
    cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_MKLDNN=ON \
        -DUSE_CUDA=ON \
        -DMXNET_CUDA_ARCH="$CI_CMAKE_CUDA_ARCH" \
        -DUSE_DIST_KVSTORE=ON\
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
        -DUSE_CPP_PACKAGE=ON \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_MKLDNN=OFF \
        -DUSE_CUDA=OFF \
        -DUSE_DIST_KVSTORE=ON \
        -DBUILD_CYTHON_MODULES=ON \
        -G Ninja /work/mxnet
    ninja
}

build_ubuntu_cpu_openblas_make() {
    set -ex
    export CC=gcc-7
    export CXX=g++-7
    build_ccache_wrappers
    make \
        DEV=1                         \
        USE_TVM_OP=1                  \
        USE_CPP_PACKAGE=1             \
        USE_BLAS=openblas             \
        USE_MKLDNN=0                  \
        USE_DIST_KVSTORE=1            \
        USE_LIBJPEG_TURBO=1           \
        USE_SIGNAL_HANDLER=1          \
        -j$(nproc)
    make cython PYTHON=python3
}

build_ubuntu_cpu_mkl() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DENABLE_TESTCOVERAGE=ON \
        -DUSE_MKLDNN=OFF \
        -DUSE_CUDA=OFF \
        -DUSE_TVM_OP=ON \
        -DUSE_MKL_IF_AVAILABLE=ON \
        -DUSE_BLAS=MKL \
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
        -G Ninja \
        /work/mxnet

    ninja
}

build_ubuntu_cpu_cmake_asan() {
    set -ex

    cd /work/build
    export CXX=g++-8
    export CC=gcc-8
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
        -DUSE_CPP_PACKAGE=ON \
        -DMXNET_USE_CPU=ON \
        /work/mxnet
    make -j $(nproc) mxnet
    # Disable leak detection but enable ASAN to link with ASAN but not fail with build tooling.
    ASAN_OPTIONS=detect_leaks=0 \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.5 \
    make -j $(nproc) mlp_cpu
}

build_ubuntu_cpu_gcc8_werror() {
    set -ex
    cd /work/build
    CXX=g++-8 CC=gcc-8 cmake \
        -DUSE_CUDA=OFF \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DUSE_CPP_PACKAGE=ON \
        -DMXNET_USE_CPU=ON \
        -GNinja /work/mxnet
    ninja
}

build_ubuntu_cpu_clang10_werror() {
    set -ex
    cd /work/build
    CXX=clang++-10 CC=clang-10 cmake \
       -DUSE_CUDA=OFF \
       -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
       -DUSE_CPP_PACKAGE=ON \
       -DMXNET_USE_CPU=ON \
       -GNinja /work/mxnet
    ninja
}

build_ubuntu_gpu_clang10_werror() {
    set -ex
    cd /work/build
    # Disable cpp package as OpWrapperGenerator.py dlopens libmxnet.so,
    # requiring presence of cuda driver libraries that are missing on CI host
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.1/targets/x86_64-linux/lib/stubs
    # Workaround https://github.com/thrust/thrust/issues/1072
    # Can be deleted on Cuda 11
    export CXXFLAGS="-I/usr/local/thrust"

    CXX=clang++-10 CC=clang-10 cmake \
       -DUSE_CUDA=ON \
       -DMXNET_CUDA_ARCH="$CI_CMAKE_CUDA_ARCH" \
       -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
       -DUSE_CPP_PACKAGE=OFF \
       -GNinja /work/mxnet
    ninja
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
        -DUSE_CPP_PACKAGE=ON \
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
       -DUSE_CPP_PACKAGE=ON \
       -G Ninja /work/mxnet
    ninja
}

build_ubuntu_cpu_clang_tidy() {
    set -ex
    cd /work/build
    export CLANG_TIDY=/usr/lib/llvm-6.0/share/clang/run-clang-tidy.py
    # TODO(leezu) USE_OPENMP=OFF 3rdparty/dmlc-core/CMakeLists.txt:79 broken?
    CXX=clang++-6.0 CC=clang-6.0 cmake \
       -DUSE_MKL_IF_AVAILABLE=OFF \
       -DUSE_MKLDNN=OFF \
       -DUSE_CUDA=OFF \
       -DUSE_OPENMP=OFF \
       -DCMAKE_BUILD_TYPE=Debug \
       -DUSE_DIST_KVSTORE=ON \
       -DUSE_CPP_PACKAGE=ON \
       -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
       -G Ninja /work/mxnet
    ninja
    cd /work/mxnet
    $CLANG_TIDY -p /work/build -j $(nproc) -clang-tidy-binary clang-tidy-6.0 /work/mxnet/src
}

build_ubuntu_cpu_clang6_mkldnn() {
    set -ex
    cd /work/build
    CXX=clang++-6.0 CC=clang-6.0 cmake \
       -DUSE_MKL_IF_AVAILABLE=OFF \
       -DUSE_MKLDNN=ON \
       -DUSE_CUDA=OFF \
       -DUSE_CPP_PACKAGE=ON \
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
       -DUSE_CPP_PACKAGE=ON \
       -G Ninja /work/mxnet
    ninja
}

build_ubuntu_cpu_mkldnn_make() {
    set -ex

    export CC=gcc-7
    export CXX=g++-7
    build_ccache_wrappers

    make  \
        DEV=1                         \
        USE_CPP_PACKAGE=1             \
        USE_TVM_OP=1                  \
        USE_BLAS=openblas             \
        USE_SIGNAL_HANDLER=1          \
        -j$(nproc)
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
        -DUSE_CPP_PACKAGE=ON \
        -G Ninja /work/mxnet
    ninja
}

build_ubuntu_cpu_mkldnn_mkl() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DENABLE_TESTCOVERAGE=ON \
        -DUSE_MKLDNN=ON \
        -DUSE_CUDA=OFF \
        -DUSE_TVM_OP=ON \
        -DUSE_MKL_IF_AVAILABLE=ON \
        -DUSE_BLAS=MKL \
        -GNinja /work/mxnet
    ninja
}

build_ubuntu_gpu() {
    build_ubuntu_gpu_cuda101_cudnn7
}

build_ubuntu_gpu_tensorrt() {

    set -ex

    export CC=gcc-7
    export CXX=g++-7

    # Build ONNX
    pushd .
    echo "Installing ONNX."
    cd 3rdparty/onnx-tensorrt/third_party/onnx
    rm -rf build
    mkdir -p build
    cd build
    cmake \
        -DCMAKE_CXX_FLAGS=-I/usr/include/python${PYVER}\
        -DBUILD_SHARED_LIBS=ON ..\
        -G Ninja
    ninja -j 1 -v onnx/onnx.proto
    ninja -j 1 -v
    export LIBRARY_PATH=`pwd`:`pwd`/onnx/:$LIBRARY_PATH
    export CPLUS_INCLUDE_PATH=`pwd`:$CPLUS_INCLUDE_PATH
    popd

    # Build ONNX-TensorRT
    pushd .
    cd 3rdparty/onnx-tensorrt/
    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc)
    export LIBRARY_PATH=`pwd`:$LIBRARY_PATH
    popd

    mkdir -p /work/mxnet/lib/
    cp 3rdparty/onnx-tensorrt/third_party/onnx/build/*.so /work/mxnet/lib/
    cp -L 3rdparty/onnx-tensorrt/build/libnvonnxparser_runtime.so.0 /work/mxnet/lib/
    cp -L 3rdparty/onnx-tensorrt/build/libnvonnxparser.so.0 /work/mxnet/lib/

    cd /work/build
    cmake -DUSE_CUDA=1                            \
          -DUSE_CUDNN=1                           \
          -DUSE_OPENCV=1                          \
          -DUSE_TENSORRT=1                        \
          -DUSE_OPENMP=0                          \
          -DUSE_MKLDNN=0                          \
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
        -DMXNET_CUDA_ARCH="$CI_CMAKE_CUDA_ARCH" \
        -DUSE_CPP_PACKAGE=ON \
        -G Ninja /work/mxnet
    ninja
}

build_ubuntu_gpu_mkldnn_nocudnn() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_CUDA=ON \
        -DMXNET_CUDA_ARCH="$CI_CMAKE_CUDA_ARCH" \
        -DUSE_CUDNN=OFF \
        -DUSE_CPP_PACKAGE=ON \
        -G Ninja /work/mxnet
    ninja
}

build_ubuntu_gpu_cuda101_cudnn7() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_CUDA=ON \
        -DMXNET_CUDA_ARCH="$CI_CMAKE_CUDA_ARCH" \
        -DUSE_CUDNN=ON \
        -DUSE_MKLDNN=OFF \
        -DUSE_CPP_PACKAGE=ON \
        -DUSE_DIST_KVSTORE=ON \
        -DBUILD_CYTHON_MODULES=ON \
        -G Ninja /work/mxnet
    ninja
}

build_ubuntu_gpu_cuda101_cudnn7_make() {
    set -ex
    export CC=gcc-7
    export CXX=g++-7
    build_ccache_wrappers
    make \
        USE_BLAS=openblas                         \
        USE_MKLDNN=0                              \
        USE_CUDA=1                                \
        USE_CUDA_PATH=/usr/local/cuda             \
        USE_CUDNN=1                               \
        USE_CPP_PACKAGE=1                         \
        USE_DIST_KVSTORE=1                        \
        CUDA_ARCH="$CI_CUDA_COMPUTE_CAPABILITIES" \
        USE_SIGNAL_HANDLER=1                      \
        -j$(nproc)
    make cython PYTHON=python3
}

build_ubuntu_gpu_cuda101_cudnn7_mkldnn_cpp_test() {
    set -ex
    export CC=gcc-7
    export CXX=g++-7
    build_ccache_wrappers
    make \
        USE_BLAS=openblas                         \
        USE_MKLDNN=1                              \
        USE_CUDA=1                                \
        USE_CUDA_PATH=/usr/local/cuda             \
        USE_CUDNN=1                               \
        USE_CPP_PACKAGE=1                         \
        USE_DIST_KVSTORE=1                        \
        CUDA_ARCH="$CI_CUDA_COMPUTE_CAPABILITIES" \
        USE_SIGNAL_HANDLER=1                      \
        -j$(nproc)
    make test USE_CPP_PACKAGE=1 -j$(nproc)
    make cython PYTHON=python3
}

build_ubuntu_amalgamation() {
    set -ex
    # Amalgamation can not be run with -j nproc
    export CC=gcc-7
    export CXX=g++-7
    build_ccache_wrappers
    make -C amalgamation/ clean
    make -C amalgamation/     \
        USE_BLAS=openblas
}

build_ubuntu_amalgamation_min() {
    set -ex
    # Amalgamation can not be run with -j nproc
    export CC=gcc-7
    export CXX=g++-7
    build_ccache_wrappers
    make -C amalgamation/ clean
    make -C amalgamation/     \
        USE_BLAS=openblas     \
        MIN=1
}

build_ubuntu_gpu_cmake() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DUSE_SIGNAL_HANDLER=ON                 \
        -DUSE_CUDA=ON                           \
        -DUSE_CUDNN=ON                          \
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
    CC=gcc-7 CXX=g++-7 cmake \
        -DUSE_SIGNAL_HANDLER=ON                 \
        -DUSE_CUDA=ON                           \
        -DUSE_CUDNN=ON                          \
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
    CC=gcc-7 CXX=g++-7 cmake \
        -DUSE_SIGNAL_HANDLER=ON                 \
        -DUSE_CUDA=OFF                          \
        -DUSE_CUDNN=OFF                         \
        -DUSE_MKLDNN=OFF                        \
        -DUSE_INT64_TENSOR_SIZE=ON              \
        -G Ninja                                \
        /work/mxnet

    ninja
}

build_ubuntu_gpu_large_tensor() {
    set -ex
    cd /work/build
    CC=gcc-7 CXX=g++-7 cmake \
        -DUSE_SIGNAL_HANDLER=ON                 \
        -DUSE_CUDA=ON                           \
        -DUSE_CUDNN=ON                          \
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

build_ubuntu_blc() {
    echo "pass"
}

# Testing

sanity_check() {
    set -ex
    tools/license_header.py check
    make cpplint jnilint
    make -f R-package/Makefile rcpplint
    make pylint
    pytest tests/tutorials/test_sanity_tutorials.py
}

# Tests libmxnet
# Parameters:
# $1 -> mxnet_variant: The variant of the libmxnet.so library
cd_unittest_ubuntu() {
    set -ex
    source /opt/rh/rh-python36/enable
    export PYTHONPATH=./python/
    export MXNET_MKLDNN_DEBUG=0  # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export MXNET_ENABLE_CYTHON=0
    export CD_JOB=1 # signal this is a CD run so any unecessary tests can be skipped
    export DMLC_LOG_STACK_TRACE_DEPTH=10

    local mxnet_variant=${1:?"This function requires a mxnet variant as the first argument"}

    pytest --durations=50 --verbose tests/python/unittest
    pytest --durations=50 --verbose tests/python/quantization

    # https://github.com/apache/incubator-mxnet/issues/11801
    # if [[ ${mxnet_variant} = "cpu" ]] || [[ ${mxnet_variant} = "mkl" ]]; then
        # integrationtest_ubuntu_cpu_dist_kvstore
    # fi

    if [[ ${mxnet_variant} = cu* ]]; then
        pytest --durations=50 --verbose tests/python/gpu

        # Adding these here as CI doesn't test all CUDA environments
        pytest example/image-classification/test_score.py
        # TODO(szha): fix and reenable the hanging issue. tracked in #18098
        # integrationtest_ubuntu_gpu_dist_kvstore
    fi

    if [[ ${mxnet_variant} = *mkl ]]; then
        pytest --durations=50 --verbose tests/python/mkl
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
    pytest --durations=50 --cov-report xml:tests_unittest.xml --verbose tests/python/unittest
    pytest --durations=50 --cov-report xml:tests_quantization.xml --verbose tests/python/quantization
}

unittest_ubuntu_python3_cpu_mkldnn() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_MKLDNN_DEBUG=0  # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export MXNET_ENABLE_CYTHON=0
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    pytest --durations=50 --cov-report xml:tests_unittest.xml --verbose tests/python/unittest
    pytest --durations=50 --cov-report xml:tests_mkl.xml --verbose tests/python/mkl
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
    pytest --durations=50 --cov-report xml:tests_gpu.xml --verbose tests/python/gpu
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
    pytest --durations=50 --cov-report xml:tests_gpu.xml --verbose tests/python/gpu
}

unittest_ubuntu_python3_gpu_nocudnn() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export CUDNN_OFF_TEST_ONLY=true
    export MXNET_ENABLE_CYTHON=0
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    pytest --durations=50 --cov-report xml:tests_gpu.xml --verbose tests/python/gpu
}

unittest_ubuntu_tensorrt_gpu() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export LD_LIBRARY_PATH=/work/mxnet/lib:$LD_LIBRARY_PATH
    export CUDNN_VERSION=${CUDNN_VERSION:-7.0.3}
    export MXNET_ENABLE_CYTHON=0
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    python3 tests/python/tensorrt/lenet5_train.py
    pytest --durations=50 --cov-report xml:tests_trt_gpu.xml --verbose --capture=no tests/python/tensorrt/
}

# quantization gpu currently only runs on P3 instances
# need to separte it from unittest_ubuntu_python3_gpu()
unittest_ubuntu_python3_quantization_gpu() {
    set -ex
    if [ -f /etc/redhat-release ]; then
        source /opt/rh/rh-python36/enable
    fi
    export PYTHONPATH=./python/
    export MXNET_MKLDNN_DEBUG=0 # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export CUDNN_VERSION=${CUDNN_VERSION:-7.0.3}
    export MXNET_ENABLE_CYTHON=0
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    pytest --durations=50 --cov-report xml:tests_quantization_gpu.xml --verbose tests/python/quantization_gpu
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
    export CC=gcc-7
    export CXX=g++-7
    build_ccache_wrappers
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
    export CC=gcc-7
    export CXX=g++-7
    build_ccache_wrappers
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
    export CC=gcc-7
    export CXX=g++-7
    build_ccache_wrappers
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
    python -m pytest --durations=50 --cov-report xml:tests_unittest.xml --verbose tests/python/unittest
    python -m pytest --durations=50 --cov-report xml:tests_train.xml --verbose tests/python/train
}

unittest_centos7_gpu() {
    set -ex
    source /opt/rh/rh-python36/enable
    cd /work/mxnet
    export CUDNN_VERSION=${CUDNN_VERSION:-7.0.3}
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    python3 -m pytest --durations=50 --cov-report xml:tests_gpu.xml --verbose tests/python/gpu
}

integrationtest_ubuntu_cpu_onnx() {
	set -ex
	export PYTHONPATH=./python/
    export DMLC_LOG_STACK_TRACE_DEPTH=10
	python3 tests/python/unittest/onnx/backend_test.py
	pytest tests/python/unittest/onnx/mxnet_export_test.py
	pytest tests/python/unittest/onnx/test_models.py
	pytest tests/python/unittest/onnx/test_node.py
}

integrationtest_ubuntu_gpu_python() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_SUBGRAPH_VERBOSE=0
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    pytest example/image-classification/test_score.py
}

integrationtest_ubuntu_cpu_asan() {
    set -ex
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
    cd /work/mxnet/python
    pip3 install -e .
    pip3 install --no-cache-dir horovod
    cd /work/mxnet/tests/nightly
    ./test_distributed_training-gpu.sh
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
    pip3 install -e .
    cd /work/mxnet
    python3 -m pytest $TEST_TIMER_ARGUMENTS --verbose tests/python/unittest

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
    python3 -m pytest --verbose tests/python/unittest/test_engine.py
}

# Functions that run the nightly Tests:

#Runs Apache RAT Check on MXNet Source for License Headers
nightly_test_rat_check() {
    set -e
    pushd .

    cd /work/deps/0.12-release/apache-rat/target

    # Use shell number 5 to duplicate the log output. It get sprinted and stored in $OUTPUT at the same time https://stackoverflow.com/a/12451419
    exec 5>&1
    OUTPUT=$(java -jar apache-rat-0.13-SNAPSHOT.jar -E /work/mxnet/tests/nightly/apache_rat_license_check/rat-excludes -d /work/mxnet|tee >(cat - >&5))
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

# Runs Imagenet inference
nightly_test_imagenet_inference() {
    set -ex
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    echo $PWD
    cp /work/mxnet/build/cpp-package/example/inference/imagenet_inference /work/mxnet/cpp-package/example/inference/
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
    pytest tests/nightly/test_large_array.py::test_tensor
    pytest tests/nightly/test_large_array.py::test_nn
    pytest tests/nightly/test_large_array.py::test_basic
}

#Test Large Vectors
nightly_test_large_vector() {
    set -ex
    export PYTHONPATH=./python/
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    pytest tests/nightly/test_large_vector.py::test_tensor
    pytest tests/nightly/test_large_vector.py::test_nn
    pytest tests/nightly/test_large_vector.py::test_basic
}

#Test Large Vectors
nightly_test_large_vector() {
    set -ex
    export PYTHONPATH=./python/
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    pytest tests/nightly/test_large_vector.py::test_tensor
    pytest tests/nightly/test_large_vector.py::test_nn
    pytest tests/nightly/test_large_vector.py::test_basic
}

#Tests Amalgamation Build with 5 different sets of flags
nightly_test_amalgamation() {
    set -ex
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    export CC=gcc-7
    export CXX=g++-7
    # Amalgamation can not be run with -j nproc
    make -C amalgamation/ clean
    make -C amalgamation/ ${1} ${2}
}

#Tests Amalgamation Build for Javascript
nightly_test_javascript() {
    set -ex
    export LLVM=/work/deps/emscripten-fastcomp/build/bin
    export DMLC_LOG_STACK_TRACE_DEPTH=10
    export CC=gcc-7
    export CXX=g++-7
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
    pytest --durations=50 --cov-report xml:tests_tutorials.xml --capture=no test_tutorials.py
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
    pytest test_estimator_cnn.py
    pytest test_sentiment_rnn.py
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

    export CC="ccache gcc"
    export CXX="ccache g++"

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
    export CC="gcc-7"
    export CXX="g++-7"
    build_ccache_wrappers
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
    api_folder='html/api'
    mkdir -p $api_folder/python/docs && tar -xzf python-artifacts.tgz --directory $api_folder/python/docs
    GZIP=-9 tar -zcvf beta_website.tgz -C html .
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
    source /opt/rh/devtoolset-7/enable
    source /opt/rh/rh-python36/enable
    export USE_SYSTEM_CUDA=1
    export CMAKE_STATICBUILD=1
    local mxnet_variant=${1:?"This function requires a python command as the first argument"}
    source tools/staticbuild/build.sh ${mxnet_variant}
    popd
}

# Packages libmxnet into wheel file
cd_package_pypi() {
    set -ex
    pushd .
    source /opt/rh/devtoolset-7/enable
    source /opt/rh/rh-python36/enable
    local mxnet_variant=${1:?"This function requires a python command as the first argument"}
    ./cd/python/pypi/pypi_package.sh ${mxnet_variant}
    popd
}

# Sanity checks wheel file
cd_integration_test_pypi() {
    set -ex
    source /opt/rh/rh-python36/enable

    local gpu_enabled=${1:-"false"}

    local test_conv_params=''
    local mnist_params=''

    if [ "${gpu_enabled}" = "true" ]; then
        mnist_params="--gpu 0"
        test_conv_params="--gpu"
    fi

    # install mxnet wheel package
    pip3 install --user ./wheel_build/dist/*.whl

    # execute tests
    python3 /work/mxnet/tests/python/train/test_conv.py ${test_conv_params}
    python3 /work/mxnet/example/image-classification/train_mnist.py ${mnist_params}
}

# Publishes wheel to PyPI
cd_pypi_publish() {
    set -ex
    pip3 install --user twine
    python3 ./cd/python/pypi/pypi_publish.py `readlink -f wheel_build/dist/*.whl`
}

cd_s3_publish() {
    set -ex
    pip3 install --user awscli
    filepath=$(readlink -f wheel_build/dist/*.whl)
    filename=$(basename $filepath)
    variant=$(echo $filename | cut -d'-' -f1 | cut -d'_' -f2 -s)
    if [ -z "${variant}" ]; then
        variant="cpu"
    fi
    aws s3 cp ${filepath} s3://apache-mxnet/dist/python/${variant}/${filename} --grants read=uri=http://acs.amazonaws.com/groups/global/AllUsers full=id=43f628fab72838a4f0b929d7f1993b14411f4b0294b011261bc6bd3e950a6822
}

build_static_scala_cpu() {
    set -ex
    pushd .
    scala_prepare
    export MAVEN_PUBLISH_OS_TYPE=linux-x86_64-cpu
    export mxnet_variant=cpu
    source /opt/rh/devtoolset-7/enable
    source /opt/rh/rh-maven35/enable
    ./ci/publish/scala/build.sh
    popd
}

build_static_python_cpu() {
    set -ex
    pushd .
    export mxnet_variant=cpu
    source /opt/rh/devtoolset-7/enable
    source /opt/rh/rh-python36/enable
    ./ci/publish/python/build.sh
    popd
}

build_static_python_cu92() {
    set -ex
    pushd .
    export mxnet_variant=cu92
    export USE_SYSTEM_CUDA=1
    source /opt/rh/devtoolset-7/enable
    source /opt/rh/rh-python36/enable
    ./ci/publish/python/build.sh
    popd
}

build_static_python_cpu_cmake() {
    set -ex
    pushd .
    export mxnet_variant=cpu
    export CMAKE_STATICBUILD=1
    source /opt/rh/devtoolset-7/enable
    source /opt/rh/rh-python36/enable
    ./ci/publish/python/build.sh
    popd
}

build_static_python_cu92_cmake() {
    set -ex
    pushd .
    export mxnet_variant=cu92
    export CMAKE_STATICBUILD=1
    export USE_SYSTEM_CUDA=1
    source /opt/rh/devtoolset-7/enable
    source /opt/rh/rh-python36/enable
    ./ci/publish/python/build.sh
    popd
}

publish_scala_build() {
    set -ex
    pushd .
    scala_prepare
    source /opt/rh/devtoolset-7/enable
    source /opt/rh/rh-maven35/enable
    export USE_SYSTEM_CUDA=1
    ./ci/publish/scala/build.sh
    popd
}

publish_scala_test() {
    set -ex
    pushd .
    scala_prepare
    source /opt/rh/rh-maven35/enable
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

# broken_link_checker

broken_link_checker() {
    set -ex
    ./tests/nightly/broken_link_checker_test/broken_link_checker.sh
}

# artifact repository unit tets
test_artifact_repository() {
    set -ex
    pushd .
    cd cd/utils/
    pytest test_artifact_repository.py
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
