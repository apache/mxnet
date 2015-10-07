#!/bin/bash

git clone --recursive https://github.com/dmlc/mxnet __mxnet_build
cd __mxnet_build

if [ ! -f config.mk ]; then
    echo "Use the default config.m"
    cp make/config.mk config.mk

    if [ ${TRAVIS_OS_NAME} == "linux" ]; then
      sed -i 's/export CC = gcc/export CC = gcc-4.8/g' config.mk
      sed -i 's/export CXX = g++/export CXX = g++-4.8/g' config.mk
      echo "=============================="
      gcc --version
      gcc-4.8 --version
    fi

    if [ ${TRAVIS_OS_NAME} == "osx" ]; then
      sed -i 's%ADD_CFLAGS =%ADD_CFLAGS = -I/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers/%' config.mk
    fi

    cat config.mk
fi

make -j4 || exit 1

export MXNET_HOME=$PWD
cd ..
