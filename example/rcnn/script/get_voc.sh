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
