#!/bin/bash

git clone --recursive https://github.com/dmlc/mxnet __mxnet_build
cd __mxnet_build

if [ ! -f config.mk ]; then
    echo "Use the default config.m"
    cp make/config.mk config.mk
fi

make -j4 || exit 1

export MXNET_HOME=$PWD
