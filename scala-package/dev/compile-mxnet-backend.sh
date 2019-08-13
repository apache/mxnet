#!/bin/bash

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# (Yizhi) This is mainly inspired by the script in apache/spark.
# I did some modificaiton to get it with our project.
#

set -e
echo "Compiling MXNet Backend, Hang tight!....."

if [[ ($# -ne 2) || ( $1 == "--help") ||  $1 == "-h" ]]; then
  echo "Usage: [-h|--help]  <osx-x86_64-cpu/linux-x86_64-cpu/linux-x86_64-gpu> <project.basedir>" 1>&2
  exit 1
fi
PLATFORM=$1
MXNETDIR=$2


# below routine shamelessly copied from
# https://github.com/apache/incubator-mxnet/blob/master/setup-utils/install-mxnet-osx-python.sh
# This routine executes a command,
# prints error message on the console on non-zero exit codes and
# returns the exit code to the caller.
chkret() {
	cmd=$*
	echo "$cmd"
	$cmd
	ret=$?
	if [[ ${ret} != 0 ]]; then
		echo " "
		echo "ERROR: Return value non-zero for: $cmd"
		echo " "
		exit 1
	fi
} # chkret()

UNAME=`uname -s`
chkret pushd $MXNETDIR

set +e
git submodule update --init --recursive
set -e

# don't want to overwrite an existing config file
cp make/config.mk ./config.mk

if [[ $PLATFORM == "osx-x86_64-cpu" ]];
then
    echo "Building MXNet Backend on MAC OS"
    echo "ADD_CFLAGS += -I/usr/local/opt/opencv/include" >> ./config.mk
    echo "ADD_CFLAGS += -I/usr/local/opt/openblas/include" >> ./config.mk
    echo "ADD_LDFLAGS += -L/usr/local/opt/opencv/lib" >> ./config.mk
    echo "ADD_LDFLAGS += -L/usr/local/opt/openblas/lib" >> ./config.mk
    echo "USE_OPENMP = 0" >> ./config.mk
    echo "USE_LAPACK_PATH = /usr/local/opt/lapack/lib" >> ./config.mk
    make -j$(sysctl -n hw.ncpu)
elif [[ $PLATFORM == "linux-x86_64-cpu" ]];
then
    echo "Building MXNet Backend on Linux CPU"
    echo "ADD_CFLAGS += -I/usr/local/include/opencv" >> ./config.mk
    echo "ADD_LDFLAGS += -L/usr/local/lib" >> ./config.mk
    echo "USE_OPENCV=1" >> ./config.mk
    echo "USE_OPENMP=1" >> ./config.mk
    echo "USE_BLAS=openblas" >> ./config.mk
    echo "USE_LAPACK=1" >> ./config.mk
    echo "USE_DIST_KVSTORE=1" >> ./config.mk
    echo "USE_S3=1" >> ./config.mk
    make -j$(cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1)
elif [[ $PLATFORM == "linux-x86_64-gpu" ]]
then
    echo "Building MXNet Backend on Linux GPU"
    echo "ADD_CFLAGS += -I/usr/local/include/opencv" >> ./config.mk
    echo "ADD_LDFLAGS += -L/usr/local/lib" >> ./config.mk
    echo "USE_OPENCV=1" >> ./config.mk
    echo "USE_OPENMP=1" >> ./config.mk
    echo "USE_BLAS=openblas" >> ./config.mk
    echo "USE_LAPACK=1" >> ./config.mk
    echo "USE_DIST_KVSTORE=1" >> ./config.mk
    echo "USE_S3=1" >> ./config.mk
    echo "USE_CUDA=1" >> ./config.mk
    echo "USE_CUDNN=1" >> ./config.mk
    echo "ADD_CFLAGS += -I/usr/local/cuda/include" >> ./config.mk
    echo "ADD_LDFLAGS += -L/usr/local/cuda/lib64/  " >> ./config.mk
    #update th nccl version approriately
    echo "ADD_LDFLAGS += -L/lib/nccl/cuda-9.0/lib  " >> ./config.mk
    eval "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/nccl/cuda-9.0/lib"
    eval "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    make -j$(cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1)
    echo "Building MXNet Backend on Linux GPU"
else
    echo "MY ALIEN OVERLOADS HAVE NOT TOLD WHAT TO DO FOR INVALID INPUT !!!"
    echo "Currently supported platforms: osx-x86_64-cpu or linux-x86_64-cpu or linux-x86_64-gpu"
fi
chkret popd
echo "done building MXNet Backend"
exit 0
