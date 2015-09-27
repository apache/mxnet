#!/bin/bash

# main script of travis
if [ ${TASK} == "lint" ]; then
    if [ ${TRAVIS_OS_NAME} != "osx" ]; then
        make lint || exit -1
    fi
    exit 0
fi

if [ ${TASK} == "doc" ]; then
    if [ ${TRAVIS_OS_NAME} != "osx" ]; then
        make doc 2>log.txt
        (cat log.txt|grep warning) && exit -1
    fi
    exit 0
fi

# prereqs for things that need make
cp make/config.mk config.mk

export NOSE3=nosetests3
export PYTHON3=python3
if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    source scripts/travis_osx_install.sh
    echo "USE_BLAS=apple" >> config.mk
    echo "USE_OPENMP=0" >> config.mk
    alias nosetests='python -m nose'
    export NOSE3='python -m nose'
    export PYTHON3=python
else
    echo "USE_BLAS=blas" >> config.mk
    echo "USE_CUDNN=0" >> config.mk
    echo "CXX=g++-4.8" >> config.mk
    export CXX="g++-4.8"
fi

echo "USE_S3=0" >> config.mk

if [ ${TASK} == "build" ]; then
    if [ ${TRAVIS_OS_NAME} != "osx" ]; then
        echo "USE_CUDA=1" >> config.mk
        echo "USE_THREADED_ENGINE=1" >> config.mk
        ./dmlc-core/scripts/setup_nvcc.sh $NVCC_PREFIX
        make all || exit -1
    fi
fi

if [ ${TASK} == "python" ]; then
    echo "USE_CUDA=0" >> config.mk
    make all || exit -1
    python --version
    export MXNET_ENGINE_TYPE=ThreadedEngine
    nosetests tests/python/unittest || exit -1
    nosetests tests/python/train || exit -1
fi

if [ ${TASK} == "python3" ]; then
    echo "USE_CUDA=0" >> config.mk
    make all || exit -1
    export MXNET_ENGINE_TYPE=ThreadedEngine
    ${PYTHON3} --version
    ${NOSE3} tests/python/unittest || exit -1
    ${NOSE3} tests/python/train || exit -1
fi

if [ ${TASK} == "python_naive" ]; then
    echo "USE_CUDA=0" >> config.mk
    make all || exit -1
    export MXNET_ENGINE_TYPE=NaiveEngine
    python --version
    nosetests tests/python/unittest || exit -1
    nosetests tests/python/train || exit -1
fi

if [ ${TASK} == "python_perdev" ]; then
    echo "USE_CUDA=0" >> config.mk
    make all || exit -1
    export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
    python --version
    nosetests tests/python/unittest || exit -1
    nosetests tests/python/train || exit -1
fi

if [ ${TASK} == "cpp_unittest" ]; then
    make -f dmlc-core/scripts/packages.mk gtest
    echo "USE_CUDA=0" >> config.mk
    echo "GTEST_PATH="${CACHE_PREFIX} >> config.mk
    make test || exit -1
    export MXNET_ENGINE_TYPE=ThreadedEngine
    export MXNET_ENGINE_INFO=true
    for test in tests/cpp/*_test; do
        ./$test || exit -1
    done
fi

