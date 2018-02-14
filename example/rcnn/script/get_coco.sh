#!/usr/bin/env bash

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


# make a data folder
if ! [ -e data ]
then
    mkdir data
fi

pushd data

# download images
mkdir images
declare -a filenames=("train2014" "val2014")
for i in "${filenames[@]}"
do
    if ! [ -e $i.zip ]
    then
        echo $i.zip "not found, downloading"
        wget http://msvocds.blob.core.windows.net/coco2014/$i.zip
    fi
    unzip $i.zip
    echo $i/*.jpg | mv -t images
    rm -r $i
done

# download annotations
anno="instances_train-val2014.zip"
if ! [ -e $anno ]
then
    echo $anno "not found, downloading"
    wget http://msvocds.blob.core.windows.net/annotations-1-0-3/$anno
fi
unzip $anno

# the result is coco/images/ coco/annotations/
mkdir coco
mv images coco
mv annotations coco

popd
