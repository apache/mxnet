#!/usr/bin/env bash

MATLAB_DIR=$(cd `dirname $0`; pwd)
DATA_DIR="${MATLAB_DIR}/data/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist, will create one";
  mkdir -p ${DATA_DIR}
fi
cd ${DATA_DIR}

# Get cat image
wget --no-check-certificate https://raw.githubusercontent.com/dmlc/mxnet.js/master/data/cat.png;

# Get inception model
wget --no-check-certificate http://data.dmlc.ml/mxnet/models/imagenet/inception-bn.tar.gz;
tar -zxvf inception-bn.tar.gz
