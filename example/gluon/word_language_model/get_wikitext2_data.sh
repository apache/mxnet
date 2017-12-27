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


RNN_DIR=$(cd `dirname $0`; pwd)
DATA_DIR="${RNN_DIR}/data/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist, will create one";
  mkdir -p ${DATA_DIR}
fi

wget -P ${DATA_DIR} https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
cd ${DATA_DIR}
unzip wikitext-2-v1.zip

# rename
mv ${DATA_DIR}/wikitext-2/wiki.test.tokens ${DATA_DIR}/wikitext-2/wiki.test.txt
mv ${DATA_DIR}/wikitext-2/wiki.valid.tokens ${DATA_DIR}/wikitext-2/wiki.valid.txt
mv ${DATA_DIR}/wikitext-2/wiki.train.tokens ${DATA_DIR}/wikitext-2/wiki.train.txt
