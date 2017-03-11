#!/bin/bash

echo "make"
cp make/config.mk .
echo "USE_CUDA=1" | tee -a config.mk
echo "USE_CUDA_PATH=/usr/local/cuda" | tee -a config.mk
echo "USE_CUDNN=1" | tee -a config.mk
echo "USE_BLAS=openblas" | tee -a config.mk
echo "EXTRA_OPERATORS=example/ssd/operator" | tee -a config.mk
make -j$(nproc) || exit -1

export PYTHONPATH=`pwd`/python/
echo $PYTHONPATH
echo "BUILD python_test"
nosetests --verbose tests/python/unittest || exit -1

echo "BUILD python3_test"
nosetests3 --verbose tests/python/unittest || exit -1

# echo "BUILD scala_test"
# export PATH=$PATH:/opt/apache-maven/bin
# make scalapkg || exit -1
# make scalatest || exit -1

# echo "BUILD julia_test"
# export MXNET_HOME="${PWD}"
# /home/ubuntu/julia/bin/julia -e 'try Pkg.clone("MXNet"); catch end; Pkg.checkout("MXNet"); Pkg.build("MXNet"); Pkg.test("MXNet")' || exit -1
