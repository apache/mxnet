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
    mv make/crosscompile.jetson.mk make/config.mk
    make -j$(nproc)

    export MXNET_LIBRARY_PATH=`pwd`/libmxnet.so
    cd /work/mxnet/python
    python setup.py bdist_wheel --universal

    # Fix pathing issues in the wheel.  We need to move libmxnet.so from the data folder to the
    # mxnet folder, then repackage the wheel.
    WHEEL=`readlink -f dist/*.whl`
    TMPDIR=`mktemp -d`
    unzip -d $TMPDIR $WHEEL
    rm $WHEEL
    cd $TMPDIR
    mv *.data/data/mxnet/libmxnet.so mxnet
    zip -r $WHEEL .
    cp $WHEEL /work/build
    rm -rf $TMPDIR
    popd
}

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
        -DCMAKE_TOOLCHAIN_FILE=$CROSS_ROOT/Toolchain.cmake \
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
        -DUSE_DIST_KVSTORE=ON\
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
        -DUSE_SSE=OFF\
        -DUSE_LAPACK=OFF\
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

build_centos7_cpu() {
    set -ex
    cd /work/mxnet
    make \
        DEV=1 \
        USE_LAPACK=1 \
        USE_LAPACK_PATH=/usr/lib64/liblapack.so \
        USE_BLAS=openblas \
        USE_DIST_KVSTORE=1 \
        -j$(nproc)
}

build_centos7_mkldnn() {
    set -ex
    cd /work/mxnet
    make \
        DEV=1 \
        USE_LAPACK=1 \
        USE_LAPACK_PATH=/usr/lib64/liblapack.so \
        USE_MKLDNN=1 \
        USE_BLAS=openblas \
        -j$(nproc)
}

build_centos7_gpu() {
    set -ex
    cd /work/mxnet
    make \
        DEV=1 \
        USE_LAPACK=1 \
        USE_LAPACK_PATH=/usr/lib64/liblapack.so \
        USE_BLAS=openblas \
        USE_CUDA=1 \
        USE_CUDA_PATH=/usr/local/cuda \
        USE_CUDNN=1 \
        USE_DIST_KVSTORE=1 \
        -j$(nproc)
}

build_ubuntu_cpu_openblas() {
    set -ex
    make \
        DEV=1                         \
        USE_CPP_PACKAGE=1             \
        USE_BLAS=openblas             \
        USE_DIST_KVSTORE=1            \
        -j$(nproc)
}

build_ubuntu_cpu_clang39() {
    set -ex
    make \
        USE_CPP_PACKAGE=1             \
        USE_BLAS=openblas             \
        USE_OPENMP=0                  \
        USE_DIST_KVSTORE=1            \
        CXX=clang++-3.9               \
        CC=clang-3.9                  \
        -j$(nproc)
}

build_ubuntu_cpu_clang50() {
    set -ex
    make \
        USE_CPP_PACKAGE=1             \
        USE_BLAS=openblas             \
        USE_OPENMP=1                  \
        USE_DIST_KVSTORE=1            \
        CXX=clang++-5.0               \
        CC=clang-5.0                  \
        -j$(nproc)
}

build_ubuntu_cpu_clang39_mkldnn() {
    set -ex
    make \
        USE_CPP_PACKAGE=1             \
        USE_BLAS=openblas             \
        USE_MKLDNN=1                  \
        USE_OPENMP=0                  \
        CXX=clang++-3.9               \
        CC=clang-3.9                  \
        -j$(nproc)
}

build_ubuntu_cpu_clang50_mkldnn() {
    set -ex
    make \
        USE_CPP_PACKAGE=1             \
        USE_BLAS=openblas             \
        USE_MKLDNN=1                  \
        USE_OPENMP=1                  \
        CXX=clang++-5.0               \
        CC=clang-5.0                  \
        -j$(nproc)
}

build_ubuntu_cpu_mkldnn() {
    set -ex
    make  \
        DEV=1                         \
        USE_CPP_PACKAGE=1             \
        USE_BLAS=openblas             \
        USE_MKLDNN=1                  \
        -j$(nproc)
}

build_ubuntu_gpu_mkldnn() {
    set -ex
    make  \
        DEV=1                         \
        USE_CPP_PACKAGE=1             \
        USE_BLAS=openblas             \
        USE_MKLDNN=1                  \
        USE_CUDA=1                    \
        USE_CUDA_PATH=/usr/local/cuda \
        USE_CUDNN=1                   \
        -j$(nproc)
}

build_ubuntu_gpu_cuda91_cudnn7() {
    set -ex
    make  \
        DEV=1                         \
        USE_BLAS=openblas             \
        USE_CUDA=1                    \
        USE_CUDA_PATH=/usr/local/cuda \
        USE_CUDNN=1                   \
        USE_CPP_PACKAGE=1             \
        USE_DIST_KVSTORE=1            \
        -j$(nproc)
}

build_ubuntu_amalgamation() {
    set -ex
    # Amalgamation can not be run with -j nproc
    make -C amalgamation/ clean
    make -C amalgamation/ USE_BLAS=openblas
}

build_ubuntu_amalgamation_min() {
    set -ex
    # Amalgamation can not be run with -j nproc
    make -C amalgamation/ clean
    make -C amalgamation/ USE_BLAS=openblas MIN=1
}

build_ubuntu_gpu_cmake_mkldnn() {
    set -ex
    cd /work/build
    cmake \
        -DUSE_CUDA=1               \
        -DUSE_CUDNN=1              \
        -DUSE_MKLML_MKL=1          \
        -DUSE_MKLDNN=1             \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja                   \
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
        -DUSE_CUDA=1               \
        -DUSE_CUDNN=1              \
        -DUSE_MKLML_MKL=0          \
        -DUSE_MKLDNN=0             \
        -DUSE_DIST_KVSTORE=1       \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja                   \
        /work/mxnet

    ninja -v
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
    # MXNET_MKLDNN_DEBUG is buggy and produces false positives
    # https://github.com/apache/incubator-mxnet/issues/10026
    #export MXNET_MKLDNN_DEBUG=1  # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    nosetests-2.7 --verbose tests/python/unittest
    nosetests-2.7 --verbose tests/python/train
    nosetests-2.7 --verbose tests/python/quantization
}

unittest_ubuntu_python3_cpu() {
    set -ex
    export PYTHONPATH=./python/
    # MXNET_MKLDNN_DEBUG is buggy and produces false positives
    # https://github.com/apache/incubator-mxnet/issues/10026
    #export MXNET_MKLDNN_DEBUG=1  # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    nosetests-3.4 --verbose tests/python/unittest
    nosetests-3.4 --verbose tests/python/quantization
}

unittest_ubuntu_python3_cpu_mkldnn() {
    set -ex
    export PYTHONPATH=./python/ 
    # MXNET_MKLDNN_DEBUG is buggy and produces false positives
    # https://github.com/apache/incubator-mxnet/issues/10026
    #export MXNET_MKLDNN_DEBUG=1  # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    nosetests-3.4 --verbose tests/python/unittest
    nosetests-3.4 --verbose tests/python/quantization
    nosetests-3.4 --verbose tests/python/mkl
}

unittest_ubuntu_python2_gpu() {
    set -ex
    export PYTHONPATH=./python/
    # MXNET_MKLDNN_DEBUG is buggy and produces false positives
    # https://github.com/apache/incubator-mxnet/issues/10026
    #export MXNET_MKLDNN_DEBUG=1  # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    nosetests-2.7 --verbose tests/python/gpu
}

tutorialtest_ubuntu_python3_gpu() {
    set -ex
    cd /work/mxnet/docs
    export MXNET_DOCS_BUILD_MXNET=0
    make html
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export PYTHONPATH=/work/mxnet/python/
    export MXNET_TUTORIAL_TEST_KERNEL=python3
    cd /work/mxnet/tests/tutorials && nosetests-3.4 test_tutorials.py --nologcapture
}

tutorialtest_ubuntu_python2_gpu() {
    set -ex
    cd /work/mxnet/docs
    export MXNET_DOCS_BUILD_MXNET=0
    make html
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    export PYTHONPATH=/work/mxnet/python/
    export MXNET_TUTORIAL_TEST_KERNEL=python2
    cd /work/mxnet/tests/tutorials && nosetests-3.4 test_tutorials.py --nologcapture
}

unittest_ubuntu_python3_gpu() {
    set -ex
    export PYTHONPATH=./python/
    # MXNET_MKLDNN_DEBUG is buggy and produces false positives
    # https://github.com/apache/incubator-mxnet/issues/10026
    #export MXNET_MKLDNN_DEBUG=1 # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    nosetests-3.4 --verbose tests/python/gpu
}

# quantization gpu currently only runs on P3 instances
# need to separte it from unittest_ubuntu_python2_gpu()
unittest_ubuntu_python2_quantization_gpu() {
    set -ex
    export PYTHONPATH=./python/
    # MXNET_MKLDNN_DEBUG is buggy and produces false positives
    # https://github.com/apache/incubator-mxnet/issues/10026
    #export MXNET_MKLDNN_DEBUG=1  # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    nosetests-2.7 --verbose tests/python/quantization_gpu
}

# quantization gpu currently only runs on P3 instances
# need to separte it from unittest_ubuntu_python3_gpu()
unittest_ubuntu_python3_quantization_gpu() {
    set -ex
    export PYTHONPATH=./python/
    # MXNET_MKLDNN_DEBUG is buggy and produces false positives
    # https://github.com/apache/incubator-mxnet/issues/10026
    #export MXNET_MKLDNN_DEBUG=1 # Ignored if not present
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    nosetests-3.4 --verbose tests/python/quantization_gpu
}

unittest_ubuntu_cpu_scala() {
    set -ex
    make scalapkg USE_BLAS=openblas USE_DIST_KVSTORE=1
    make scalatest USE_BLAS=openblas USE_DIST_KVSTORE=1
}

unittest_ubuntu_gpu_scala() {
    set -ex
    make scalapkg USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 USE_DIST_KVSTORE=1
    make scalatest USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 SCALA_TEST_ON_GPU=1 USE_DIST_KVSTORE=1
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
    # make -j not supported
    make rpkg USE_BLAS=openblas R_LIBS=/tmp/r-site-library
    R CMD INSTALL --library=/tmp/r-site-library R-package
    make rpkgtest R_LIBS=/tmp/r-site-library
}

unittest_ubuntu_gpu_R() {
    set -ex
    mkdir -p /tmp/r-site-library
    # make -j not supported
    make rpkg USE_BLAS=openblas R_LIBS=/tmp/r-site-library
    R CMD INSTALL --library=/tmp/r-site-library R-package
    make rpkgtest R_LIBS=/tmp/r-site-library R_GPU_ENABLE=1
}

unittest_centos7_cpu() {
    set -ex
    cd /work/mxnet
    python3.6 -m "nose" --with-timer --verbose tests/python/unittest
    python3.6 -m "nose" --with-timer --verbose tests/python/train
}

unittest_centos7_gpu() {
    set -ex
    cd /work/mxnet
    python3.6 -m "nose" --with-timer --verbose tests/python/gpu
}

integrationtest_ubuntu_cpu_onnx() {
	set -ex
	export PYTHONPATH=./python/
	python example/onnx/super_resolution.py
	pytest tests/python-pytest/onnx/onnx_backend_test.py
	pytest tests/python-pytest/onnx/onnx_test.py
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

integrationtest_ubuntu_gpu_cpp_package() {
    set -ex
    cpp-package/tests/ci_test.sh
}

integrationtest_ubuntu_gpu_dist_kvstore() {
    set -ex
    export PYTHONPATH=./python/
    export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
    cd tests/nightly/
    ../../tools/launch.py -n 7 --launcher local python dist_sync_kvstore.py
    ../../tools/launch.py -n 7 --launcher local python dist_sync_kvstore.py --no-multiprecision
    ../../tools/launch.py -n 7 --launcher local python dist_device_sync_kvstore.py
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

build_docs() {
    set -ex
    pushd .
    cd /work/mxnet/docs/build_version_doc
    ./build_all_version.sh $1
    ./update_all_version.sh $2 $3 $4
    cd VersionedWeb
    tar -zcvf ../artifacts.tgz .
    popd
}

# Deploy

deploy_docs() {
    set -ex
    pushd .

    make docs

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
