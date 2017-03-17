#!/bin/bash

echo "BUILD make"
cp make/config.mk .
echo "USE_CUDA=1" >> config.mk
echo "USE_CUDA_PATH=/usr/local/cuda" >> config.mk
echo "USE_CUDNN=1" >> config.mk
echo "USE_PROFILER=1" >> config.mk
echo "DEV=1" >> config.mk
echo "EXTRA_OPERATORS=example/ssd/operator" >> config.mk
user=`id -u -n`

set -e

make -j$(nproc) || exit 1

export PYTHONPATH=${PWD}/python

echo "BUILD python_test"
nosetests --verbose tests/python/unittest || exit 1
nosetests --verbose tests/python/gpu/test_operator_gpu.py || exit 1
nosetests --verbose tests/python/gpu/test_forward.py || exit 1
nosetests --verbose tests/python/train || exit 1

echo "BUILD python3_test"
nosetests3 --verbose tests/python/unittest || exit 1
nosetests3 --verbose tests/python/gpu/test_operator_gpu.py || exit 1
nosetests3 --verbose tests/python/gpu/test_forward.py || exit 1
nosetests3 --verbose tests/python/train || exit 1

echo "BUILD scala_test"
export PATH=$PATH:/opt/apache-maven/bin
make scalapkg || exit 1
make scalatest || exit 1

