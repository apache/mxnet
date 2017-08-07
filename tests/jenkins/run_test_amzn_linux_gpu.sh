#!/bin/bash

# Exit script with error if any errors occur

echo "BUILD make"
cp make/config.mk .
echo "USE_CUDA=0" >> config.mk
echo "USE_CUDNN=0" >> config.mk
echo "USE_BLAS=openblas" >> config.mk
echo "USE_CPP_PACKAGE=1" >> config.mk
echo "ADD_CFLAGS += -I/usr/include/openblas" >>config.mk
echo "GTEST_PATH=/usr/local/gtest" >> config.mk
echo 'export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH' >> ~/.profile
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.profile
JAVA_HOME=`/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.*.amzn1.x86_64[-1]`
echo 'export JAVA_HOME=${JAVA_HOME}' >> ~/.profile
echo 'export JRE_HOME=${JAVA_HOME}/jre' >> ~/.profile
echo 'export PATH=$PATH:/apache-maven-3.3.9/bin/:/usr/bin:${JAVA_HOME}/bin' >> ~/.profile
source ~/.profile
user=`id -u -n`

set -e

make -j 4

echo "BUILD cpp_test"
make -j 4 test
export MXNET_ENGINE_INFO=true
./build/tests/cpp/mxnet_test

echo "BUILD valgrind_test"
valgrind ./build/tests/cpp/mxnet_test

export MXNET_ENGINE_INFO=false
export PYTHONPATH=${PWD}/python

echo "BUILD python_test"
nosetests --verbose tests/python/unittest
nosetests --verbose tests/python/train

echo "BUILD python3_test"
nosetests3 --verbose tests/python/unittest
nosetests3 --verbose tests/python/train

#echo "BUILD julia_test"
#export MXNET_HOME="${PWD}"
#julia -e 'try Pkg.clone("MXNet"); catch end; Pkg.checkout("MXNet"); Pkg.build("MXNet"); Pkg.test("MXNet")' || exit -1

echo "BUILD scala_test"
make scalapkg
make scalatest
