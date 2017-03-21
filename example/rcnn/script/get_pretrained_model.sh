#!/usr/bin/env bash

# make a model folder
if ! [ -e model ]
then
    mkdir model
fi

# download pretrained model
pushd model
wget http://data.dmlc.ml/mxnet/models/imagenet/vgg/vgg16-0000.params
wget http://data.dmlc.ml/mxnet/models/imagenet/resnet/101-layers/resnet-101-0000.params
popd
