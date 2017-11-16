#!/usr/bin/env bash
# libraries for building mxnet c++ core on ubuntu

apt-get update && apt-get install -y \
    build-essential git libatlas-base-dev libopencv-dev python-opencv \
    libcurl4-openssl-dev libgtest-dev cmake wget unzip

cd /usr/src/gtest && cmake CMakeLists.txt && make && cp *.a /usr/lib
