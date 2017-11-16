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

mnist_data_path="$data_path/mnist.zip"
if [ ! -f "$mnist_data_path" ]; then
  wget http://data.mxnet.io/mxnet/data/mnist.zip -P $data_path
  cd $data_path
  unzip -u mnist.zip
fi
