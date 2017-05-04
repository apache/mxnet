#!/usr/bin/env bash
# install libraries for mxnet's perl package on ubuntu
apt-get update && apt-get install -y libmouse-perl pdl cpanminus swig libgraphviz-perl
cpanm -q Function::Parameters
