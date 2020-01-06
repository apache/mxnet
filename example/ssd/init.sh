#!/bin/bash

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

#
# Train SSD with Pascal VOC
#
set -e
set -x


function download_pascal_voc() {
    pushd .
    cd data
    wget -c --show-progress http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    wget -c --show-progress http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget -c --show-progress http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    tar -xvf VOCtrainval_11-May-2012.tar
    tar -xvf VOCtrainval_06-Nov-2007.tar
    tar -xvf VOCtest_06-Nov-2007.tar
    popd
}

function download_model() {
    pushd .
    MODELFILEURL="https://github.com/zhreshold/mxnet-ssd/releases/download/v0.6/resnet50_ssd_512_voc0712_trainval.zip"
    MODELFILE="resnet50_ssd_512_voc0712_trainval.zip"
    cd model
    wget -c $MODELFILEURL
    unzip $MODELFILE
    popd
}

function download_demo_images() {
    pushd .
    cd data/demo
    ./download_demo_images.py
    popd
}

# Uncomment to download training dataset
#download_pascal_voc
download_model
download_demo_images
