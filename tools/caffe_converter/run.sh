#!/bin/bash
if [[ $# -ne 1 ]]; then
    echo "usage: $0 model_name"
    echo "   model_name: [vgg16|vgg19], ..."
    exit -1
fi

if [[ $1 == "vgg19" ]]; then
    if [[ ! -f VGG_ILSVRC_19_layers_deploy.prototxt ]]; then
        wget -c https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/bb2b4fe0a9bb0669211cf3d0bc949dfdda173e9e/VGG_ILSVRC_19_layers_deploy.prototxt
    fi

    if [[ ! -f VGG_ILSVRC_19_layers.caffemodel ]]; then
        wget -c http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel
    fi

    echo "converting"
    python `dirname $0`/convert_model.py VGG_ILSVRC_19_layers_deploy.prototxt VGG_ILSVRC_19_layers.caffemodel vgg19
elif [[ $1 == "vgg16" ]]; then
    if [[ ! -f VGG_ILSVRC_16_layers_deploy.prototxt ]]; then
        wget -c https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/c3ba00e272d9f48594acef1f67e5fd12aff7a806/VGG_ILSVRC_16_layers_deploy.prototxt
    fi

    if [[ ! -f VGG_ILSVRC_16_layers.caffemodel ]]; then
        wget -c http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
    fi

    echo "converting"
    python `dirname $0`/convert_model.py VGG_ILSVRC_16_layers_deploy.prototxt VGG_ILSVRC_16_layers.caffemodel vgg16
else
    echo "unsupported model: $1"
fi
