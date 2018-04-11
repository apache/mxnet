#!/usr/bin/env bash

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
set -ex

apt-get install -y \
    libgflags-dev \
    libgoogle-glog-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libprotobuf-dev \
    libsnappy-dev \
    protobuf-compiler \
    python-dev \
    python-numpy \
    python-opencv

apt-get install -y --no-install-recommends libboost-all-dev

cd /work/deps
git clone http://github.com/BVLC/caffe.git

cd caffe
cp Makefile.config.example Makefile.config

echo "CPU_ONLY := 1" >> Makefile.config

# Fixes https://github.com/BVLC/caffe/issues/5658 See https://github.com/intel/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide
echo "INCLUDE_DIRS += /usr/lib /usr/lib/x86_64-linux-gnu /usr/include/hdf5/serial/ " >> Makefile.config
echo "LIBRARY_DIRS += /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial " >> Makefile.config

# Fixes https://github.com/BVLC/caffe/issues/4333 See https://github.com/intel/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide
# Note: This is only valid on Ubuntu16.04 - the version numbers are bound to the distribution
ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial.so.10.0.2 /usr/lib/x86_64-linux-gnu/libhdf5.so
ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial_hl.so.10.0.2 /usr/lib/x86_64-linux-gnu/libhdf5_hl.so

make all pycaffe -j$(nproc)

cd python
for req in $(cat requirements.txt); do pip2 install $req; done
