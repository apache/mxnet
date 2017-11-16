#!/bin/bash

set -e

if [ ! -z "$MXNET_DATA_DIR" ]; then
  data_path="$MXNET_DATA_DIR"
else
  data_path="./data"
fi

if [ ! -d "$data_path" ]; then
  mkdir -p "$data_path"
fi

cifar_data_path="$data_path/cifar10.zip"
if [ ! -f "$cifar_data_path" ]; then
  wget http://data.mxnet.io/mxnet/data/cifar10.zip -P $data_path
  cd $data_path
  unzip -u cifar10.zip
fi
