#!/bin/bash

set -e

echo "BUILD make"

WITH_CAFFE_PLUGIN=0

if [ "$WITH_CAFFE_PLUGIN" == "1" ]; then
# Check out caffe
  git clone https://github.com/BVLC/caffe
  mkdir -p caffe/build
  cd caffe/build
  cmake ..
  make -j$(nproc)
  cd ../..
fi

cp make/config.mk .
echo "USE_CUDA=1" >> config.mk
echo "USE_CUDA_PATH=/usr/local/cuda" >> config.mk
echo "USE_CUDNN=1" >> config.mk
echo "USE_PROFILER=1" >> config.mk
echo "DEV=1" >> config.mk
echo "EXTRA_OPERATORS=example/ssd/operator" >> config.mk
echo "USE_CPP_PACKAGE=1" >> config.mk

if [ "$WITH_CAFFE_PLUGIN" == "1" ]; then
    echo "CAFFE_PATH = $(pwd)/caffe" >> config.mk
    echo "MXNET_PLUGINS += plugin/caffe/caffe.mk" >> config.mk
fi

user=`id -u -n`

make -j$(nproc)

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

