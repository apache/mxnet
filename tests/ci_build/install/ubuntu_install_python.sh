#!/usr/bin/env bash
# install libraries for mxnet's python package on ubuntu

apt-get update && apt-get install -y \
    python-pip python-dev python-numpy \
    python3-pip python3-dev python3-numpy

pip install nose pylint
pip3 install nose pylint
