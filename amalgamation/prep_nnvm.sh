#! /bin/bash
DMLC_CORE=$(pwd)/../dmlc-core
cd ../nnvm/amalgamation
make clean
make DMLC_CORE_PATH=$DMLC_CORE nnvm.d
cp nnvm.d ../../amalgamation/
echo '#define MSHADOW_FORCE_STREAM

#ifndef MSHADOW_USE_CBLAS
#if (__MIN__ == 1)
#define MSHADOW_USE_CBLAS   0
#else
#define MSHADOW_USE_CBLAS   1
#endif
#endif
#define MSHADOW_USE_CUDA    0
#define MSHADOW_USE_MKL     0
#define MSHADOW_RABIT_PS    0
#define MSHADOW_DIST_PS     0
#define DMLC_LOG_STACK_TRACE 0

#include "mshadow/tensor.h"
#include "mxnet/base.h"
#include "dmlc/json.h"
#include "nnvm/tuple.h"
#include "mxnet/tensor_blob.h"' > temp
cat nnvm.cc >> temp
mv temp ../../amalgamation/nnvm.cc
