#!/usr/bin/env bash

# make a data folder
if ! [ -e data ]
then
    mkdir data
fi

pushd data

# the result is selective_search_data
wget http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/selective_search_data.tgz
tar xf selective_search_data.tgz

popd
