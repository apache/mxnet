#!/usr/bin/env bash

# make a data folder
if ! [ -e data ]
then
    mkdir data
fi

pushd data

# the result is VOCdevkit/VOC2007
declare -a arr=("VOCtrainval_06-Nov-2007.tar" "VOCtest_06-Nov-2007.tar")
for i in "${arr[@]}"
do
    if ! [ -e $i ]
    then
        echo $i "not found, downloading"
        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/$i
    fi
    tar -xf $i
done

# the result is VOCdevkit/VOC2012
voc2012="VOCtrainval_11-May-2012.tar"
if ! [ -e $voc2012 ]
then
    echo $voc2012 "not found, downloading"
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/$voc2012
fi
tar -xf $voc2012

popd
