#!/bin/sh

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

set -e

### Install git
apt-get update
apt-get install git-all

### Build MXNet with CPU support
echo "BUILD make"
cp ./make/config.mk .
echo "USE_CUDA=1" >> ./config.mk
echo "USE_CUDA_PATH=/usr/local/cuda" >> config.mk
echo "USE_CUDNN=1" >> ./config.mk
echo "USE_BLAS=openblas" >> ./config.mk
echo "ADD_CFLAGS += -I/usr/include/openblas" >> ./config.mk
echo "GTEST_PATH=/usr/local/gtest" >> ./config.mk
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.111-1.b15.25.amzn1.x86_64
export JRE_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.111-1.b15.25.amzn1.x86_64/jre
export PATH=$PATH:/apache-maven-3.3.9/bin/:/usr/bin:/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.111-1.b15.25.amzn1.x86_64/bin:/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.111-1.b15.25.amzn1.x86_64/jre/bin

make clean
make -j 4 || exit -1

echo "BUILD python2 mxnet"
cd ./python
python setup.py install || exit 1

echo "BUILD python3 mxnet"
python3 setup.py install || exit 1

# Come out of MXNet directory
cd ..

# Dependencies required for Keras installation
pip install pyyaml

pip install --upgrade pip
pip install --upgrade six

# If already exist remove and fork DMLC/keras and install.
# Note: This should eventually be replaced with pip install when mxnet backend is part of fchollet/keras

########### Set up Keras ####################
echo "Installing Keras. This can take few minutes..."
# Clone keras repository from dmlc. This has mxnet backend implementated.
if [ -d "keras" ]; then
  rm -rf keras/
fi

git clone https://github.com/dmlc/keras.git --recursive
cd keras
python setup.py install

########### Set up packages for profiling #########
echo "Installing memory_profile and psutil for profiling. This can take few minutes..."
pip install memory_profiler
pip install psutil

########## Set Environment Variables ########
echo "Setting Environment Variables for MXNet Keras Integration Tests on CPU machine"
cd ../../mxnet/tests/nightly

export KERAS_BACKEND="mxnet"
export MXNET_KERAS_TEST_MACHINE='GPU'
########## Call the test script with 1 GPUS ############

export GPU_NUM='1'
echo "Running MXNet Keras Integration Test on GPU machine with 1 GPUs"
nosetests --with-xunit --quiet --nologcapture mxnet_keras_integration_tests/

########## Call the test script with 2 GPUS ############

export GPU_NUM='2'
echo "Running MXNet Keras Integration Test on GPU machine with 2 GPUs"
nosetests --with-xunit --quiet --nologcapture mxnet_keras_integration_tests/

########## Call the test script with 4 GPUS ############

export GPU_NUM='4'
echo "Running MXNet Keras Integration Test on GPU machine with 4 GPUs"
nosetests --with-xunit --quiet --nologcapture mxnet_keras_integration_tests/

########## Call the test script with 8 GPUS ############

export GPU_NUM='8'
echo "Running MXNet Keras Integration Test on GPU machine with 8 GPUs"
nosetests --with-xunit --quiet --nologcapture mxnet_keras_integration_tests/
