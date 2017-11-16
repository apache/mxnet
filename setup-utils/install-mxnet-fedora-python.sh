#!/usr/bin/env bash
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
