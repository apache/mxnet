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

echo
echo "NOTE: To continue, you need to review the licensing of the data sets used by this script"
echo "See https://catalog.ldc.upenn.edu/ldc99t42 for the licensing"
read -p "Please confirm you have reviewed the licensing [Y/n]:" -n 1 -r
echo

if [ $REPLY != "Y" ]
then
    echo "License was not reviewed, aborting script."
    exit 1
fi

RNN_DIR=$(cd `dirname $0`; pwd)
DATA_DIR="${RNN_DIR}/data/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist, will create one";
  mkdir -p ${DATA_DIR}
fi

wget -P ${DATA_DIR} https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/ptb/ptb.train.txt;
wget -P ${DATA_DIR} https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/ptb/ptb.valid.txt;
wget -P ${DATA_DIR} https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/ptb/ptb.test.txt;
wget -P ${DATA_DIR} https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tinyshakespeare/input.txt;
