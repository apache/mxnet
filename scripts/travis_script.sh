#!/bin/bash

# main script of travis
if [ ${TASK} == "lint" ]; then
    make lint || exit -1
    exit 0
fi

if [ ${TASK} == "doc" ]; then
    make doc 2>log.txt
    (cat log.txt|grep warning) && exit -1
    exit 0
fi

# prereqs for things that need make
cp make/config.mk config.mk
echo "USE_BLAS=blas" >> config.mk
echo "USE_CUDNN=0" >> config.mk
echo "CXX=g++-4.8" >> config.mk
export CXX="g++-4.8"


if [ ${TASK} == "build" ]; then
    echo "USE_CUDA=1" >> config.mk
    echo "USE_THREADED_ENGINE=1" >> config.mk
    ./dmlc-core/scripts/setup_nvcc.sh $NVCC_PREFIX
    make all || exit -1
fi

if [ ${TASK} == "python" ]; then
    echo "USE_CUDA=0" >> config.mk
    echo "USE_THREADED_ENGINE=1" >> config.mk
    make all || exit -1
    nosetests tests/python || exit -1
fi

if [ ${TASK} == "python3" ]; then
    echo "USE_CUDA=0" >> config.mk
    echo "USE_THREADED_ENGINE=1" >> config.mk
    make all || exit -1
    nosetests3 tests/python || exit -1
fi

if [ ${TASK} == "python_naive" ]; then
    echo "USE_CUDA=0" >> config.mk
    make all || exit -1
    nosetests tests/python || exit -1
fi

# TODO(yutian): add unittest back
