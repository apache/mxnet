#!/usr/bin/env bash
# install libraries for building mxnet c++ core on ubuntu

apt-get update && apt-get install -y \
    build-essential git libopenblas-dev liblapack-dev libopencv-dev \
    libcurl4-openssl-dev libgtest-dev cmake wget unzip

cd /usr/src/gtest && cmake CMakeLists.txt && make && cp *.a /usr/lib
