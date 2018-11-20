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
CI_CUDA_COMPUTE_CAPABILITIES="-gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_70,code=sm_70"
CI_CMAKE_CUDA_ARCH_BIN="52,70"

clean_repo() {
    set -ex
    git clean -xfd
    git submodule foreach --recursive git clean -xfd
    git reset --hard
    git submodule foreach --recursive git reset --hard
    git submodule update --init --recursive
}

build_ccache_wrappers() {
    set -ex

    rm -f cc
    rm -f cxx

    touch cc
    touch cxx

    if [ -z ${CC+x} ]; then
        echo "No \$CC set, defaulting to gcc";
        export CC=gcc
    fi

    if [ -z ${CXX+x} ]; then
       echo "No \$CXX set, defaulting to g++";
       export CXX=g++
    fi

    # this function is nessesary for cuda enabled make based builds, since nvcc needs just an executable for -ccbin

    echo -e "#!/bin/sh\n/usr/local/bin/ccache ${CC} \"\$@\"\n" >> cc
    echo -e "#!/bin/sh\n/usr/local/bin/ccache ${CXX} \"\$@\"\n" >> cxx

    chmod +x cc
    chmod +x cxx

    export CC=`pwd`/cc
    export CXX=`pwd`/cxx
}

build_wheel() {

    set -ex
    pushd .

    PYTHON_DIR=${1:-/work/mxnet/python}
    BUILD_DIR=${2:-/work/build}

    # build

    export MXNET_LIBRARY_PATH=${BUILD_DIR}/libmxnet.so

    cd ${PYTHON_DIR}
    python setup.py bdist_wheel --universal

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

# Build commands: Every platform in docker/Dockerfile.build.<platform> should have a corresponding
# function here with the same suffix:

build_jetson() {
    set -ex
    pushd .

    cp make/crosscompile.jetson.mk ./config.mk
    make -j$(nproc)

    build_wheel /work/mxnet/python /work/mxnet/lib
    popd
}

#
# ARM builds
#

build_armv6() {
    set -ex
    pushd .
    cd /work/build

    # Lapack functionality will be included and statically linked to openblas.
    # But USE_LAPACK needs to be set to OFF, otherwise the main CMakeLists.txt
    # file tries to add -llapack. Lapack functionality though, requires -lgfortran
    # to be linked additionally.

    # We do not need OpenMP, since most armv6 systems have only 1 core

    cmake \
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DUSE_CUDA=OFF \
        -DUSE_OPENCV=OFF \
        -DUSE_OPENMP=OFF \
        -DUSE_SIGNAL_HANDLER=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_LAPACK=OFF \
        -DBUILD_CPP_EXAMPLES=OFF \
        -Dmxnet_LINKER_LIBS=-lgfortran \
        -G Ninja /work/mxnet

    ninja -v
    build_wheel
    popd
}

build_armv7() {
    set -ex
    pushd .
    cd /work/build

    # Lapack functionality will be included and statically linked to openblas.
    # But USE_LAPACK needs to be set to OFF, otherwise the main CMakeLists.txt
    # file tries to add -llapack. Lapack functionality though, requires -lgfortran
    # to be linked additionally.

    cmake \
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
        -DCMAKE_CROSSCOMPILING=ON \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DUSE_CUDA=OFF \
        -DUSE_OPENCV=OFF \
        -DUSE_OPENMP=ON \
        -DUSE_SIGNAL_HANDLER=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_LAPACK=OFF \
        -DBUILD_CPP_EXAMPLES=OFF \
        -Dmxnet_LINKER_LIBS=-lgfortran \
        -G Ninja /work/mxnet

    ninja -v
    build_wheel
    popd
}

build_armv8() {
    cmake \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DUSE_CUDA=OFF\
        -DSUPPORT_F16C=OFF\
        -DUSE_OPENCV=OFF\
        -DUSE_OPENMP=ON \
        -DUSE_LAPACK=OFF\
        -DUSE_SIGNAL_HANDLER=ON\
        -DCMAKE_BUILD_TYPE=Release\
        -DUSE_MKL_IF_AVAILABLE=OFF\
        -G Ninja /work/mxnet
    ninja -v
    build_wheel
}


#
# ANDROID builds
#

build_android_armv7() {
    set -ex
    cd /work/build
    cmake \
        -DANDROID=ON\
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DUSE_CUDA=OFF\
        -DUSE_SSE=OFF\
        -DSUPPORT_F16C=OFF\
        -DUSE_LAPACK=OFF\
        -DUSE_OPENCV=OFF\
        -DUSE_OPENMP=OFF\
        -DUSE_SIGNAL_HANDLER=ON\
        -DCMAKE_BUILD_TYPE=RelWithDebInfo\
        -DUSE_MKL_IF_AVAILABLE=OFF\
        -G Ninja /work/mxnet
    ninja -v
}

build_android_armv8() {
    set -ex
    cd /work/build
    cmake\
        -DANDROID=ON \
        -DUSE_CUDA=OFF\
        -DUSE_SSE=OFF\
        -DUSE_LAPACK=OFF\
        -DUSE_OPENCV=OFF\
        -DUSE_OPENMP=OFF\
        -DUSE_SIGNAL_HANDLER=ON\
        -DCMAKE_BUILD_TYPE=RelWithDebInfo\
        -DUSE_MKL_IF_AVAILABLE=OFF\
        -G Ninja /work/mxnet
    ninja -v
}

build_centos7_cpu() {
    set -ex
    cd /work/mxnet
    export CC="ccache gcc"
    export CXX="ccache g++"

    make \
        DEV=1 \
        USE_LAPACK=1 \
        ENABLE_TESTCOVERAGE=1 \
        USE_LAPACK_PATH=/usr/lib64/liblapack.so \
        USE_BLAS=openblas \
        USE_DIST_KVSTORE=1 \
        -j$(nproc)
}

build_amzn_linux_cpu() {
    cd /work/build
    cmake \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DENABLE_TESTCOVERAGE=ON \
        -DUSE_CUDA=OFF\
        -DUSE_OPENCV=ON\
        -DUSE_OPENMP=ON\
        -DUSE_SIGNAL_HANDLER=ON\
        -DCMAKE_BUILD_TYPE=RelWithDebInfo\
        -DUSE_MKL_IF_AVAILABLE=OFF\
        -DUSE_LAPACK=OFF\
        -DUSE_DIST_KVSTORE=ON\
        -G Ninja /work/mxnet
    ninja -v
}


build_centos7_mkldnn() {
    set -ex
    cd /work/mxnet
    export CC="ccache gcc"
    export CXX="ccache g++"

    make \
        DEV=1 \
        ENABLE_TESTCOVERAGE=1 \
        USE_LAPACK=1 \
        USE_LAPACK_PATH=/usr/lib64/liblapack.so \
        USE_MKLDNN=1 \
        USE_BLAS=openblas \
        -j$(nproc)
}

build_centos7_gpu() {
    set -ex
    cd /work/mxnet
    # unfortunately this build has problems in 3rdparty dependencies with ccache and make
    # build_ccache_wrappers
    make \
        DEV=1                                     \
        ENABLE_TESTCOVERAGE=1                     \
        USE_LAPACK=1                              \
        USE_LAPACK_PATH=/usr/lib64/liblapack.so   \
        USE_BLAS=openblas                         \
        USE_CUDA=1                                \
        USE_CUDA_PATH=/usr/local/cuda             \
        USE_CUDNN=1                               \
        USE_DIST_KVSTORE=1                        \
        CUDA_ARCH="$CI_CUDA_COMPUTE_CAPABILITIES" \
        -j$(nproc)
}

build_ubuntu_cpu() {
    build_ubuntu_cpu_openblas
}

build_ubuntu_cpu_openblas() {
    set -ex
    export CC="ccache gcc"
    export CXX="ccache g++"
    make \
        DEV=1                         \
        ENABLE_TESTCOVERAGE=1         \
        USE_CPP_PACKAGE=1             \
        USE_BLAS=openblas             \
        USE_DIST_KVSTORE=1            \
        -j$(nproc)
}

build_ubuntu_cpu_cmake_debug() {
    set -ex
    pushd .
    cd /work/build
    cmake \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DENABLE_TESTCOVERAGE=ON \
        -DUSE_CUDA=OFF \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_OPENMP=OFF \
        -DUSE_OPENCV=ON \
        -DCMAKE_BUILD_TYPE=Debug \
        -G Ninja \
        /work/mxnet

    ninja -v
    popd
}

build_ubuntu_cpu_cmake_asan() {
    set -ex

    pushd .
    cd /work/build
    cmake \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER=/usr/bin/g++-8 \
        -DCMAKE_C_COMPILER=/usr/bin/gcc-8 \
        -DUSE_CUDA=OFF \
        -DUSE_MKL_IF_AVAILABLE=OFF \
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
    popd
}

build_ubuntu_cpu_clang39() {
    set -ex
     export CXX=clang++-3.9
    export CC=clang-3.9
     build_ccache_wrappers
     make \
        ENABLE_TESTCOVERAGE=1         \
        USE_CPP_PACKAGE=1             \
        USE_BLAS=openblas             \
        USE_OPENMP=0                  \
        USE_DIST_KVSTORE=1            \
        -j$(nproc)
}

build_ubuntu_cpu_clang60() {
    set -ex

    export CXX=clang++-6.0
    export CC=clang-6.0

    build_ccache_wrappers

    make  \
        ENABLE_TESTCOVERAGE=1         \
        USE_CPP_PACKAGE=1             \
        USE_BLAS=openblas             \
        USE_OPENMP=1                  \
        USE_DIST_KVSTORE=1            \
        -j$(nproc)
}

build_ubuntu_cpu_clang_tidy() {
    set -ex

    export CXX=clang++-6.0
    export CC=clang-6.0
    export CLANG_TIDY=/usr/lib/llvm-6.0/share/clang/run-clang-tidy.py

    pushd .
    cd /work/build
    cmake \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DUSE_CUDA=OFF \
        -DUSE_MKL_IF_AVAILABLE=OFF \
        -DUSE_OPENCV=ON \
        -DCMAKE_BUILD_TYPE=Debug \
        -G Ninja \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        /work/mxnet

    ninja -v
    cd /work/mxnet
    $CLANG_TIDY -p /work/build -j $(nproc) -clang-tidy-binary clang-tidy-6.0 /work/mxnet/src
    popd
}

build_ubuntu_cpu_clang39_mkldnn() {
    set -ex

    export CXX=clang++-3.9
    export CC=clang-3.9

    build_ccache_wrappers

    make \
        ENABLE_TESTCOVERAGE=1         \
        USE_CPP_PACKAGE=1             \
        USE_BLAS=openblas             \
        USE_MKLDNN=1                  \
        USE_OPENMP=0                  \
        -j$(nproc)
}

build_ubuntu_cpu_clang60_mkldnn() {
    set -ex

    export CXX=clang++-6.0
    export CC=clang-6.0

    build_ccache_wrappers

    make \
        ENABLE_TESTCOVERAGE=1         \
        USE_CPP_PACKAGE=1             \
        USE_BLAS=openblas             \
        USE_MKLDNN=1                  \
        USE_OPENMP=1                  \
        -j$(nproc)
}

build_ubuntu_cpu_mkldnn() {
    set -ex

    build_ccache_wrappers

    make  \
        DEV=1                         \
        ENABLE_TESTCOVERAGE=1         \
        USE_CPP_PACKAGE=1             \
        USE_BLAS=openblas             \
        USE_MKLDNN=1                  \
        -j$(nproc)
}

build_ubuntu_gpu() {
    build_ubuntu_gpu_cuda91_cudnn7
}

build_ubuntu_gpu_tensorrt() {

    set -ex

    build_ccache_wrappers

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

    rm -rf build
    make \
        DEV=1                                                \
        ENABLE_TESTCOVERAGE=1                                \
        USE_BLAS=openblas                                    \
        USE_CUDA=1                                           \
        USE_CUDA_PATH=/usr/local/cuda                        \
        USE_CUDNN=1                                          \
        USE_OPENCV=0                                         \
        USE_DIST_KVSTORE=0                                   \
        USE_TENSORRT=1                                       \
        USE_JEMALLOC=0                                       \
        USE_GPERFTOOLS=0                                     \
        ONNX_NAMESPACE=onnx                                  \
        CUDA_ARCH="-gencode arch=compute_70,code=compute_70" \
        -j$(nproc)
}

build_ubuntu_gpu_mkldnn() {
    set -ex

    build_ccache_wrappers

    make  \
        DEV=1                                     \
        ENABLE_TESTCOVERAGE=1                     \
        USE_CPP_PACKAGE=1                         \
        USE_BLAS=openblas                         \
        USE_MKLDNN=1                              \
        USE_CUDA=1                                \
        USE_CUDA_PATH=/usr/local/cuda             \
        USE_CUDNN=1                               \
        CUDA_ARCH="$CI_CUDA_COMPUTE_CAPABILITIES" \
        -j$(nproc)
}

build_ubuntu_gpu_mkldnn_nocudnn() {
    set -ex

    build_ccache_wrappers

    make  \
        DEV=1                                     \
        ENABLE_TESTCOVERAGE=1                     \
        USE_BLAS=openblas                         \
        USE_MKLDNN=1                              \
        USE_CUDA=1                                \
        USE_CUDA_PATH=/usr/local/cuda             \
        USE_CUDNN=0                               \
        CUDA_ARCH="$CI_CUDA_COMPUTE_CAPABILITIES" \
        -j$(nproc)
}

build_ubuntu_gpu_cuda91_cudnn7() {
    set -ex
    # unfortunately this build has problems in 3rdparty dependencies with ccache and make
    # build_ccache_wrappers
    make \
        DEV=1                                     \
        ENABLE_TESTCOVERAGE=1                     \
        USE_BLAS=openblas                         \
        USE_CUDA=1                                \
        USE_CUDA_PATH=/usr/local/cuda             \
        USE_CUDNN=1                               \
        USE_CPP_PACKAGE=1                         \
        USE_DIST_KVSTORE=1                        \
        CUDA_ARCH="$CI_CUDA_COMPUTE_CAPABILITIES" \
        -j$(nproc)
}

build_ubuntu_amalgamation() {
    set -ex
    # Amalgamation can not be run with -j nproc
    make -C amalgamation/ clean
    make -C amalgamation/     \
        USE_BLAS=openblas     \
        ENABLE_TESTCOVERAGE=1
}

build_ubuntu_amalgamation_min() {
    set -ex
    # Amalgamation can not be run with -j nproc
    make -C amalgamation/ clean
    make -C amalgamation/     \
        USE_BLAS=openblas     \
        MIN=1                 \
        ENABLE_TESTCOVERAGE=1
}

build_ubuntu_gpu_cmake_mkldnn() {
    set -ex
    cd /work/build
    cmake \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache    \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache      \
        -DENABLE_TESTCOVERAGE=ON                \
        -DUSE_CUDA=1                            \
        -DUSE_CUDNN=1                           \
        -DUSE_MKLML_MKL=1                       \
        -DUSE_MKLDNN=1                          \
        -DCMAKE_BUILD_TYPE=Release              \
        -DCUDA_ARCH_NAME=Manual                 \
        -DCUDA_ARCH_BIN=$CI_CMAKE_CUDA_ARCH_BIN \
        -G Ninja                                \
        /work/mxnet

    ninja -v
    # libmkldnn.so.0 is a link file. We need an actual binary file named libmkldnn.so.0.
    cp 3rdparty/mkldnn/src/libmkldnn.so.0 3rdparty/mkldnn/src/libmkldnn.so.0.tmp
    mv 3rdparty/mkldnn/src/libmkldnn.so.0.tmp 3rdparty/mkldnn/src/libmkldnn.so.0
}

build_ubuntu_gpu_cmake() {
    set -ex
    cd /work/build
    cmake \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache    \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache      \
        -DENABLE_TESTCOVERAGE=ON                \
        -DUSE_CUDA=1                            \
        -DUSE_CUDNN=1                           \
        -DUSE_MKLML_MKL=0                       \
        -DUSE_MKLDNN=0                          \
        -DUSE_DIST_KVSTORE=1                    \
        -DCMAKE_BUILD_TYPE=Release              \
        -DCUDA_ARCH_NAME=Manual                 \
        -DCUDA_ARCH_BIN=$CI_CMAKE_CUDA_ARCH_BIN \
        -G Ninja                                \
        /work/mxnet

    ninja -v
}

build_ubuntu_blc() {
    echo "pass"
}

# Testing

sanity_check() {
    set -ex
    tools/license_header.py check
    make cpplint rcpplint jnilint
    make pylint
    nosetests-3.4 tests/tutorials/test_sanity_tutorials.py
}


unittest_ubuntu_python2_cpu() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_MKLDNN_DEBUG=1
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    nosetests-2.7 $NOSE_COVERAGE_ARGUMENTS --with-xunit --xunit-file nosetests_unittest.xml --verbose tests/python/unittest
    nosetests-2.7 $NOSE_COVERAGE_ARGUMENTS --with-xunit --xunit-file nosetests_train.xml --verbose tests/python/train
    nosetests-2.7 $NOSE_COVERAGE_ARGUMENTS --with-xunit --xunit-file nosetests_quantization.xml --verbose tests/python/quantization
}

unittest_ubuntu_python3_cpu() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_MKLDNN_DEBUG=1  # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    nosetests-3.4 $NOSE_COVERAGE_ARGUMENTS --with-xunit --xunit-file nosetests_unittest.xml --verbose tests/python/unittest
    nosetests-3.4 $NOSE_COVERAGE_ARGUMENTS --with-xunit --xunit-file nosetests_quantization.xml --verbose tests/python/quantization
}

unittest_ubuntu_python3_cpu_mkldnn() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_MKLDNN_DEBUG=1  # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    nosetests-3.4 $NOSE_COVERAGE_ARGUMENTS --with-xunit --xunit-file nosetests_unittest.xml --verbose tests/python/unittest
    nosetests-3.4 $NOSE_COVERAGE_ARGUMENTS --with-xunit --xunit-file nosetests_mkl.xml --verbose tests/python/mkl
}

unittest_ubuntu_python2_gpu() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_MKLDNN_DEBUG=1  # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export CUDNN_VERSION=7.0.3
    nosetests-2.7 $NOSE_COVERAGE_ARGUMENTS --with-xunit --xunit-file nosetests_gpu.xml --verbose tests/python/gpu
}

unittest_ubuntu_python3_gpu() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_MKLDNN_DEBUG=1 # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export CUDNN_VERSION=7.0.3
    nosetests-3.4 $NOSE_COVERAGE_ARGUMENTS --with-xunit --xunit-file nosetests_gpu.xml --verbose tests/python/gpu
}

unittest_ubuntu_python3_gpu_nocudnn() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export CUDNN_OFF_TEST_ONLY=true
    nosetests-3.4 $NOSE_COVERAGE_ARGUMENTS --with-xunit --xunit-file nosetests_gpu.xml --verbose tests/python/gpu
}

unittest_ubuntu_tensorrt_gpu() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export LD_LIBRARY_PATH=/work/mxnet/lib:$LD_LIBRARY_PATH
    export CUDNN_VERSION=7.0.3
    python tests/python/tensorrt/lenet5_train.py
    nosetests-3.4 $NOSE_COVERAGE_ARGUMENTS --with-xunit --xunit-file nosetests_trt_gpu.xml --verbose --nocapture tests/python/tensorrt/
}

# quantization gpu currently only runs on P3 instances
# need to separte it from unittest_ubuntu_python2_gpu()
unittest_ubuntu_python2_quantization_gpu() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_MKLDNN_DEBUG=1  # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export CUDNN_VERSION=7.0.3
    nosetests-2.7 $NOSE_COVERAGE_ARGUMENTS --with-xunit --xunit-file nosetests_quantization_gpu.xml --verbose tests/python/quantization_gpu
}

# quantization gpu currently only runs on P3 instances
# need to separte it from unittest_ubuntu_python3_gpu()
unittest_ubuntu_python3_quantization_gpu() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_MKLDNN_DEBUG=1 # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export CUDNN_VERSION=7.0.3
    nosetests-3.4 $NOSE_COVERAGE_ARGUMENTS --with-xunit --xunit-file nosetests_quantization_gpu.xml --verbose tests/python/quantization_gpu
}

unittest_ubuntu_cpu_scala() {
    set -ex
    make scalapkg USE_BLAS=openblas USE_DIST_KVSTORE=1 ENABLE_TESTCOVERAGE=1
    make scalaunittest USE_BLAS=openblas USE_DIST_KVSTORE=1 ENABLE_TESTCOVERAGE=1
}

unittest_centos7_cpu_scala() {
    set -ex
    cd /work/mxnet
    make scalapkg USE_BLAS=openblas USE_DIST_KVSTORE=1 ENABLE_TESTCOVERAGE=1
    make scalaunittest USE_BLAS=openblas USE_DIST_KVSTORE=1 ENABLE_TESTCOVERAGE=1
}

unittest_ubuntu_cpu_clojure() {
    set -ex
    make scalapkg USE_OPENCV=1 USE_BLAS=openblas USE_DIST_KVSTORE=1 ENABLE_TESTCOVERAGE=1
    make scalainstall USE_OPENCV=1 USE_BLAS=openblas USE_DIST_KVSTORE=1 ENABLE_TESTCOVERAGE=1
    ./contrib/clojure-package/ci-test.sh
}

unittest_ubuntu_cpugpu_perl() {
    set -ex
    ./perl-package/test.sh
}

unittest_ubuntu_gpu_cpp() {
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
    make rpkg                           \
        USE_BLAS=openblas               \
        R_LIBS=/tmp/r-site-library

    R CMD INSTALL --library=/tmp/r-site-library R-package
    make rpkgtest R_LIBS=/tmp/r-site-library
}

unittest_ubuntu_gpu_R() {
    set -ex
    mkdir -p /tmp/r-site-library
    # build R packages in parallel
    mkdir -p ~/.R/
    echo  "MAKEFLAGS = -j"$(nproc) > ~/.R/Makevars
    # make -j not supported
    make rpkg                           \
        USE_BLAS=openblas               \
        R_LIBS=/tmp/r-site-library
    R CMD INSTALL --library=/tmp/r-site-library R-package
    make rpkgtest R_LIBS=/tmp/r-site-library R_GPU_ENABLE=1
}

unittest_ubuntu_cpu_julia06() {
    set -ex
    export PATH="/work/julia/bin:$PATH"
    export MXNET_HOME='/work/mxnet'
    export JULIA_PKGDIR='/work/julia-pkg'
    export DEPDIR=`julia -e 'print(Pkg.dir())'`

    julia -e 'versioninfo()'
    julia -e 'Pkg.init()'

    # install package
    ln -sf ${MXNET_HOME}/julia ${DEPDIR}/MXNet

    # install dependencies
    julia -e 'Pkg.resolve()'

    # FIXME
    export LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libjemalloc.so'

    # use the prebuilt binary from $MXNET_HOME/lib
    julia -e 'Pkg.build("MXNet")'

    # run the script `julia/test/runtests.jl`
    julia -e 'Pkg.test("MXNet")'

    # See https://github.com/dmlc/MXNet.jl/pull/303#issuecomment-341171774
    julia -e 'using MXNet; mx._sig_checker()'
}

unittest_centos7_cpu() {
    set -ex
    cd /work/mxnet
    python3.6 -m "nose" $NOSE_COVERAGE_ARGUMENTS --with-xunit --xunit-file nosetests_unittest.xml --verbose tests/python/unittest
    python3.6 -m "nose" $NOSE_COVERAGE_ARGUMENTS --with-xunit --xunit-file nosetests_train.xml --verbose tests/python/train
}

unittest_centos7_gpu() {
    set -ex
    cd /work/mxnet
    export CUDNN_VERSION=7.0.3
    python3.6 -m "nose" $NOSE_COVERAGE_ARGUMENTS --with-xunit --xunit-file nosetests_gpu.xml --verbose tests/python/gpu
}

integrationtest_ubuntu_cpu_onnx() {
	set -ex
	export PYTHONPATH=./python/
	pytest tests/python-pytest/onnx/import/mxnet_backend_test.py
	pytest tests/python-pytest/onnx/import/onnx_import_test.py
	pytest tests/python-pytest/onnx/import/gluon_backend_test.py
	pytest tests/python-pytest/onnx/export/onnx_backend_test.py
	python tests/python-pytest/onnx/export/mxnet_export_test.py
}

integrationtest_ubuntu_gpu_python() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    python example/image-classification/test_score.py
}

integrationtest_ubuntu_gpu_caffe() {
    set -ex
    export PYTHONPATH=/work/deps/caffe/python:./python
    python tools/caffe_converter/test_converter.py
}

integrationtest_ubuntu_cpu_asan() {
    set -ex
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.5

    # We do not want to fail the build on ASAN errors until memory leaks have been addressed.
    export ASAN_OPTIONS=exitcode=0
    cd /work/mxnet/build/cpp-package/example/
    /work/mxnet/cpp-package/example/get_data.sh
    ./mlp_cpu
}

integrationtest_ubuntu_gpu_cpp_package() {
    set -ex
    cpp-package/tests/ci_test.sh
}

integrationtest_ubuntu_cpu_dist_kvstore() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export MXNET_USE_OPERATOR_TUNING=0
    cd tests/nightly/
    ../../tools/launch.py -n 7 --launcher local python dist_sync_kvstore.py --type=gluon_step_cpu
    ../../tools/launch.py -n 7 --launcher local python dist_sync_kvstore.py --type=gluon_sparse_step_cpu
    ../../tools/launch.py -n 7 --launcher local python dist_sync_kvstore.py --type=invalid_cpu
    ../../tools/launch.py -n 7 --launcher local python dist_sync_kvstore.py --type=gluon_type_cpu
    ../../tools/launch.py -n 7 --launcher local python dist_sync_kvstore.py
    ../../tools/launch.py -n 7 --launcher local python dist_sync_kvstore.py --no-multiprecision
    ../../tools/launch.py -n 7 --launcher local python dist_sync_kvstore.py --type=compressed_cpu
    ../../tools/launch.py -n 7 --launcher local python dist_sync_kvstore.py --type=compressed_cpu --no-multiprecision
    ../../tools/launch.py -n 3 --launcher local python test_server_profiling.py
}

integrationtest_ubuntu_gpu_scala() {
    set -ex
    make scalapkg USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 USE_DIST_KVSTORE=1 SCALA_ON_GPU=1 ENABLE_TESTCOVERAGE=1
    make scalaintegrationtest USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 SCALA_TEST_ON_GPU=1 USE_DIST_KVSTORE=1 ENABLE_TESTCOVERAGE=1
}

integrationtest_ubuntu_gpu_dist_kvstore() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    cd tests/nightly/
    ../../tools/launch.py -n 7 --launcher local python dist_device_sync_kvstore.py
    ../../tools/launch.py -n 7 --launcher local python dist_sync_kvstore.py --type=init_gpu
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
    python -m "nose" $NOSE_COVERAGE_ARGUMENTS --with-timer --verbose tests/python/unittest
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
    python3 -m "nose" $NOSE_COVERAGE_ARGUMENTS --with-timer --verbose tests/python/unittest

    popd
}

build_docs() {
    set -ex
    pushd .
    cd /work/mxnet/docs/build_version_doc
    # Parameters are set in the Jenkins pipeline: restricted-website-build
    # $1: the list of branches/tags to build
    # $2: the list of tags to display
    # So you can build from the 1.2.0 branch, but display 1.2.1 on the site
    # $3: the fork URL
    ./build_all_version.sh $1 $2 $3
    # $4: the default version tag for the website
    # $5: the base URL
    ./update_all_version.sh $2 $4 $5
    cd VersionedWeb
    tar -zcvf ../artifacts.tgz .
    popd
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

#Runs a simple MNIST training example
nightly_test_image_classification() {
    set -ex
    ./tests/nightly/test_image_classification.sh
}

#Single Node KVStore Test
nightly_test_KVStore_singleNode() {
    set -ex
    export PYTHONPATH=./python/
    python tests/nightly/test_kvstore.py
}

#Tests Amalgamation Build with 5 different sets of flags
nightly_test_amalgamation() {
    set -ex
    # Amalgamation can not be run with -j nproc
    make -C amalgamation/ clean
    make -C amalgamation/ ${1} ${2}
}

#Tests Amalgamation Build for Javascript
nightly_test_javascript() {
    set -ex
    export LLVM=/work/deps/emscripten-fastcomp/build/bin
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
    ./tests/nightly/model_backwards_compatibility_check/model_backward_compat_checker.sh
}

#Backfills S3 bucket with models trained on earlier versions of mxnet
nightly_model_backwards_compat_train() {
    set -ex
    export PYTHONPATH=./python/
    ./tests/nightly/model_backwards_compatibility_check/train_mxnet_legacy_models.sh
}

# Nightly 'MXNet: The Straight Dope' Single-GPU Tests
nightly_straight_dope_python2_single_gpu_tests() {
    set -ex
    cd /work/mxnet/tests/nightly/straight_dope
    export PYTHONPATH=/work/mxnet/python/
    export MXNET_TEST_KERNEL=python2
    nosetests-2.7 --with-xunit --xunit-file nosetests_straight_dope_python2_single_gpu.xml \
      test_notebooks_single_gpu.py --nologcapture
}

nightly_straight_dope_python3_single_gpu_tests() {
    set -ex
    cd /work/mxnet/tests/nightly/straight_dope
    export PYTHONPATH=/work/mxnet/python/
    export MXNET_TEST_KERNEL=python3
    nosetests-3.4 --with-xunit --xunit-file nosetests_straight_dope_python3_single_gpu.xml \
      test_notebooks_single_gpu.py --nologcapture
}

# Nightly 'MXNet: The Straight Dope' Multi-GPU Tests
nightly_straight_dope_python2_multi_gpu_tests() {
    set -ex
    cd /work/mxnet/tests/nightly/straight_dope
    export PYTHONPATH=/work/mxnet/python/
    export MXNET_TEST_KERNEL=python2
    nosetests-2.7 --with-xunit --xunit-file nosetests_straight_dope_python2_multi_gpu.xml \
      test_notebooks_multi_gpu.py --nologcapture
}

nightly_straight_dope_python3_multi_gpu_tests() {
    set -ex
    cd /work/mxnet/tests/nightly/straight_dope
    export PYTHONPATH=/work/mxnet/python/
    export MXNET_TEST_KERNEL=python3
    nosetests-3.4 --with-xunit --xunit-file nosetests_straight_dope_python3_multi_gpu.xml \
      test_notebooks_multi_gpu.py --nologcapture
}

nightly_tutorial_test_ubuntu_python3_gpu() {
    set -ex
    cd /work/mxnet/docs
    export BUILD_VER=tutorial 
    export MXNET_DOCS_BUILD_MXNET=0
    make html
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export PYTHONPATH=/work/mxnet/python/
    export MXNET_TUTORIAL_TEST_KERNEL=python3
    cd /work/mxnet/tests/tutorials
    nosetests-3.4 --with-xunit --xunit-file nosetests_tutorials.xml test_tutorials.py --nologcapture
}

nightly_tutorial_test_ubuntu_python2_gpu() {
    set -ex
    cd /work/mxnet/docs
    export BUILD_VER=tutorial
    export MXNET_DOCS_BUILD_MXNET=0
    make html
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export PYTHONPATH=/work/mxnet/python/
    export MXNET_TUTORIAL_TEST_KERNEL=python2
    cd /work/mxnet/tests/tutorials
    nosetests-3.4 --with-xunit --xunit-file nosetests_tutorials.xml test_tutorials.py --nologcapture
}


# Deploy

deploy_docs() {
    set -ex
    pushd .

    make docs

    popd
}

deploy_jl_docs() {
    set -ex
    export PATH="/work/julia/bin:$PATH"
    export MXNET_HOME='/work/mxnet'
    export JULIA_PKGDIR='/work/julia-pkg'
    export DEPDIR=`julia -e 'print(Pkg.dir())'`

    julia -e 'versioninfo()'
    julia -e 'Pkg.init()'
    ln -sf ${MXNET_HOME}/julia ${DEPDIR}/MXNet
    julia -e 'Pkg.resolve()'

    # FIXME
    export LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libjemalloc.so'

    # use the prebuilt binary from $MXNET_HOME/lib
    julia -e 'Pkg.build("MXNet")'
    # build docs
    julia -e 'Pkg.add("Documenter")'
    julia -e 'cd(Pkg.dir("MXNet")); include(joinpath("docs", "make.jl"))'

    # TODO: make Jenkins worker push to MXNet.jl ph-pages branch if master build
    # ...
}

# broken_link_checker

broken_link_checker() {
    set -ex
    ./tests/nightly/broken_link_checker_test/broken_link_checker.sh
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


