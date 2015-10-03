#!/bin/bash

cp ../lib/libmxnet.a ./mxnet/src
cp ../dmlc-core/libdmlc.a mxnet/src

mkdir -p ./mxnet/inst
mkdir -p ./mxnet/inst/include

cp ../include/mxnet/c_api.h ./mxnet/inst/include

R CMD INSTALL mxnet
