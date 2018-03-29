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

######################################################################
# This script installs MXNet for Python in a virtualenv on OSX and ubuntu
######################################################################
set -e
#set -x

BUILDIR=build
VENV=mxnet_py3

setup_virtualenv() {
    if [ ! -d $VENV ];then
        virtualenv -p `which python3` $VENV
    fi
    source $VENV/bin/activate
}

gpu_count() {
    nvidia-smi -L | wc -l
}

detect_platform() {
	unameOut="$(uname -s)"
	case "${unameOut}" in
		Linux*)
			distro=$(awk -F= '/^NAME/{gsub(/"/, "", $2); print $2}' /etc/os-release)
			machine="Linux/$distro"
		;;
		Darwin*)    machine=Mac;;
		CYGWIN*)    machine=Cygwin;;
		MINGW*)     machine=MinGw;;
		*)          machine="UNKNOWN:${unameOut}"
	esac
	echo ${machine}
}


if [ $(gpu_count) -ge 1 ];then
    USE_CUDA=ON
else
    USE_CUDA=OFF
fi

PLATFORM=$(detect_platform)
echo "Detected platform '$PLATFORM'"

if [ $PLATFORM = "Mac" ];then
    USE_OPENMP=OFF
else
    USE_OPENMP=ON
fi

if [ $PLATFORM = "Linux/Ubuntu" ];then
    install_dependencies_ubuntu() {
        sudo apt-get update
        sudo apt-get install -y build-essential libatlas-base-dev libopencv-dev graphviz virtualenv cmake\
            ninja-build libopenblas-dev liblapack-dev python3 python3-dev
    }
    echo "Installing build dependencies in Ubuntu!"
    install_dependencies_ubuntu
fi

echo "Preparing a Python virtualenv in ${VENV}"
setup_virtualenv

echo "Building MXNet core. This can take a few minutes..."
build_mxnet() {
    pushd .
    set -x
    mkdir -p $BUILDIR && cd $BUILDIR
    cmake -DUSE_CUDA=$USE_CUDA -DUSE_OPENCV=ON -DUSE_OPENMP=$USE_OPENMP -DUSE_SIGNAL_HANDLER=ON -DCMAKE_BUILD_TYPE=Release -GNinja ..  
    ninja
    set +x
    popd
}


build_mxnet

echo "Installing mxnet under virtualenv ${VENV}"
install_mxnet() {
    pushd .
    cd python
    pip3 install -e .
    pip3 install opencv-python matplotlib graphviz jupyter ipython
    popd
}

install_mxnet

echo "

========================================================================================
Done! MXNet for Python installation is complete. Go ahead and explore MXNet with Python.
========================================================================================

Use the following command to enter the virtualenv:
$ source ${VENV}/bin/activate
$ iptyhon

You can then start using mxnet

import mxnet as mx
x = mx.nd.ones((5,5))
"
