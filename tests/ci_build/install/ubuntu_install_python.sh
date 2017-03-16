#!/usr/bin/env bash
# install libraries for mxnet's python package on ubuntu

apt-get update && apt-get install -y python-dev python3-dev

# the version of the pip shipped with ubuntu may be too lower, install a recent version here
cd /tmp && wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && python2 get-pip.py

pip2 install nose pylint numpy nose-timer requests
pip3 install nose pylint numpy nose-timer requests
