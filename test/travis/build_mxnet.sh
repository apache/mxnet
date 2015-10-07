#!/bin/bash

git clone --recursive https://github.com/dmlc/mxnet
cd mxnet

if [ ! -f config.mk ]; then
    echo "Use the default config.m"
    cp make/config.mk config.mk
fi

make -j4
export MXNET_HOME=$PWD
