#!/usr/bin/env bash

# download and build openblas
echo "Building mkldnn..."
curl -s -L https://github.com/01org/mkl-dnn/releases/download/v$MKLDNN_VERSION/mklml_lnx_$MKLML_VERSION.tgz -o $DEPS_PATH/mklml.tgz

# prepare MKL
tar -xzf $DEPS_PATH/mklml.tgz -C $DEPS_PATH
cp $DEPS_PATH/mklml_lnx_$MKLML_VERSION/include/* $DEPS_PATH/include
cp $DEPS_PATH/mklml_lnx_$MKLML_VERSION/lib/* $DEPS_PATH/lib
cp $DEPS_PATH/mklml_lnx_$MKLML_VERSION/license.txt $DEPS_PATH/mklml_license.txt

# build MKLDNN
curl -s -L https://github.com/01org/mkl-dnn/archive/v$MKLDNN_VERSION.zip -o $DEPS_PATH/mkldnn.zip
unzip -q $DEPS_PATH/mkldnn.zip -d $DEPS_PATH
mkdir $DEPS_PATH/mkl-dnn-$MKLDNN_VERSION/build
cd $DEPS_PATH/mkl-dnn-$MKLDNN_VERSION/build
cmake -q \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=$DEPS_PATH .. || exit 1;

make --quiet -j $NUM_PROC || exit 1;
make test || exit 1;
make install;
cd -;

export USE_MKLML=1
