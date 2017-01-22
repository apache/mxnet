#! /bin/bash
cd ../nnvm/amalgamation
make clean
make nnvm.d
cp nnvm.d ../../amalgamation/
echo '
#include "mshadow/tensor.h"
#include "mxnet/base.h"
#include "dmlc/json.h"
#include "nnvm/tuple.h"
#include "mxnet/tensor_blob.h"' > temp
cat nnvm.cc >> temp
mv temp ../../amalgamation/nnvm.cc
