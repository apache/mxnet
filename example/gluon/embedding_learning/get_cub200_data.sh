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


EMB_DIR=$(cd `dirname $0`; pwd)
DATA_DIR="${EMB_DIR}/data/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist, will create one.";
  mkdir -p ${DATA_DIR}
fi

# the dataset is from Caltech-UCSD Birds 200
# http://www.vision.caltech.edu/visipedia/CUB-200.html
# These datasets are copyright Caltech Computational Vision Group and licensed CC BY 4.0 Attribution.
# See http://www.vision.caltech.edu/archive.html for details
wget -P ${DATA_DIR} http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
cd ${DATA_DIR}; tar -xf CUB_200_2011.tgz
