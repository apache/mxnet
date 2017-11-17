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
### Build MXNet with CPU support
echo "BUILD make"
cp ./make/config.mk .
echo "USE_CUDA=0" >> ./config.mk
echo "USE_CUDNN=0" >> ./config.mk
echo "USE_BLAS=openblas" >> ./config.mk
echo "ADD_CFLAGS += -I/usr/include/openblas" >> ./config.mk
echo "GTEST_PATH=/usr/local/gtest" >> ./config.mk
echo 'export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH' >> ~/.profile
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.profile
echo 'export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.111-1.b15.25.amzn1.x86_64' >> ~/.profile
echo 'export JRE_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.111-1.b15.25.amzn1.x86_64/jre' >> ~/.profile
echo 'export PATH=$PATH:/apache-maven-3.3.9/bin/:/usr/bin:/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.111-1.b15.25.amzn1.x86_64/bin:/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.111-1.b15.25.amzn1.x86_64/jre/bin' >> ~/.profile
source ~/.profile
make clean
make -j 4 || exit -1

echo "BUILD python2 mxnet"
cd ./python
python setup.py install || exit 1

echo "BUILD python3 mxnet"
python3 setup.py install || exit 1

# Come out of Mxnet directory.
cd ..

# Required for Keras installation
pip install pyyaml

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
export KERAS_BACKEND="mxnet"
export MXNET_KERAS_TEST_MACHINE='CPU'

########## Call the test script ############
cd ../../mxnet/tests/nightly
echo "Running MXNet Keras Integration Test on CPU machine"
nosetests --with-xunit --quiet --nologcapture mxnet_keras_integration_tests/
