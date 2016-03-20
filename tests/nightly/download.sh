#!/bin/bash

dmlc_download() {
    url=http://data.dmlc.ml/mxnet/datasets/
    dir=$1
    file=$2
    if [ ! -e data/${dir}/$file ]; then
        wget ${url}/${dir}/${file} -P data/${dir}/ || exit -1
    else
        echo "data/${dir}/$file already exits"
    fi
}

dmlc_download mnist t10k-images-idx3-ubyte
dmlc_download mnist t10k-labels-idx1-ubyte
dmlc_download mnist train-images-idx3-ubyte
dmlc_download mnist train-labels-idx1-ubyte

dmlc_download cifar10 train.rec
dmlc_download cifar10 test.rec
