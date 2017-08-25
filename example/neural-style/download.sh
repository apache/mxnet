#!/bin/bash

mkdir -p model
cd model
wget https://github.com/dmlc/web-data/raw/master/mxnet/neural-style/model/vgg19.params
cd ..

mkdir -p input
cd input
wget https://github.com/dmlc/web-data/raw/master/mxnet/neural-style/input/IMG_4343.jpg
wget https://github.com/dmlc/web-data/raw/master/mxnet/neural-style/input/starry_night.jpg
cd ..

mkdir -p output
