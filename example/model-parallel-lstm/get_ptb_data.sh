#!/usr/bin/env bash

RNN_DIR=$(cd `dirname $0`; pwd)
DATA_DIR="${RNN_DIR}/data/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist, will create one";
  mkdir -p ${DATA_DIR}
fi

wget -P ${DATA_DIR} https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/ptb/ptb.train.txt;
wget -P ${DATA_DIR} https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/ptb/ptb.valid.txt;
wget -P ${DATA_DIR} https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/ptb/ptb.test.txt;
wget -P ${DATA_DIR} https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tinyshakespeare/input.txt;
