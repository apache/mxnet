#!/bin/bash

  # - export PYTHONPATH=${PYTHONPATH}:${PWD}/python
python --version
${CXX} --version

echo ${TASK}
echo ${TRAVIS_OS_NAME}
echo ${CXX}

if [ ${TASK} == "lint" ]; then
    make lint
    exit $?
fi

if [ ${TASK} == "doc" ]; then
    make doc | tee 2>log.txt
    (cat log.txt|grep warning) && exit -1
    exit 0
fi

# more setups

# prereqs for things that need make
cp make/config.mk config.mk


if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    echo "USE_BLAS=apple" >> config.mk
    echo "USE_OPENMP=0" >> config.mk
else
    # use g++-4.8 for linux
    if [ ${CXX} == "g++" ]; then
        export CXX=g++-4.8
    fi
    echo "USE_BLAS=blas" >> config.mk
fi
echo "CXX=${CXX}" >>config.mk

if [ ${TASK} == "build" ]; then
    if [ ${TRAVIS_OS_NAME} == "linux" ]; then
        echo "USE_CUDA=1" >> config.mk
        ./dmlc-core/scripts/setup_nvcc.sh $NVCC_PREFIX
    fi
    make all
    exit $?
fi

if [ ${TASK} == "cpp_test" ]; then
    make -f dmlc-core/scripts/packages.mk gtest
    echo "GTEST_PATH="${CACHE_PREFIX} >> config.mk
    make test || exit -1
    export MXNET_ENGINE_INFO=true
    for test in tests/cpp/*_test; do
        ./$test || exit -1
    done
    exit 0
fi

if [ ${TASK} == "python_test" ]; then
    make all || exit -1
    # use cached dir for storing data
    rm -rf tests/python/data
    mkdir -p ${CACHE_PREFIX}/data
    ln -s ${CACHE_PREFIX}/data tests/python/data

    python --version
    if [ ${TRAVIS_OS_NAME} == "osx" ]; then
        alias nosetests='python -m nose'
    fi
    python python/setup.py develop --user
    nosetests tests/python/unittest || exit -1
    nosetests tests/python/train || exit -1

    if [ ${TRAVIS_OS_NAME} == "linux" ]; then
        python3 --version
        rm -rf python/mxnet/mxnet.egg-info
        python3 python/setup.py develop --user
        nosetests3 tests/python/unittest || exit -1
        nosetests3 tests/python/train || exit -1
    fi
fi

exit 0

export NOSE3=nosetests3
export PYTHON3=python3
if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    source scripts/travis_osx_install.sh
    alias nosetests='python -m nose'
    export NOSE3='python -m nose'
    export PYTHON3=python
else
    echo "USE_CUDNN=0" >> config.mk
    echo "CXX=g++-4.8" >> config.mk
    export CXX="g++-4.8"
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
