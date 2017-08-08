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
# This script installs MXNet for Python along with all required dependencies on a Fedora Machine.
# Tested on Fedora 21.0 + distro.
######################################################################
set -e

MXNET_HOME="$HOME/mxnet/"
echo "MXNet root folder: $MXNET_HOME"

echo "Installing basic development tools, atlas, opencv, pip, graphviz ..."
sudo yum update
sudo yum groupinstall -y "Development Tools" "Development Libraries"
sudo yum install -y atlas atlas-devel opencv opencv-devel graphviz graphviz-devel

echo "Building MXNet core. This can take few minutes..."
cd "$MXNET_HOME"
cp make/config.mk .
make -j$(nproc)

echo "Installing Numpy..."
sudo yum install numpy

echo "Installing Python setuptools..."
sudo yum install -y python-setuptools python-pip

echo "Adding MXNet path to your ~/.bashrc file"
echo "export PYTHONPATH=$MXNET_HOME/python:$PYTHONPATH" >> ~/.bashrc
source ~/.bashrc

echo "Install Graphviz for plotting MXNet network graph..."
sudo pip install graphviz

echo "Installing Jupyter notebook..."
sudo pip install jupyter

echo "Done! MXNet for Python installation is complete. Go ahead and explore MXNet with Python :-)"
