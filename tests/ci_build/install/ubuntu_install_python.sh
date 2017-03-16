#!/usr/bin/env bash
# install libraries for mxnet's python package on ubuntu

apt-get update && apt-get install -y \
    python-pip python-dev \
    python3-pip python3-dev

pip install nose pylint numpy nose-timer
pip3 install nose pylint numpy nose-timer
