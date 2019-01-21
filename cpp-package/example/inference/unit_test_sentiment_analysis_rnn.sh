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


set -e # exit on the first error
export EXE_NAME=sentiment_analysis_rnn

# Running the example with dog image.
if [ "$(uname)" == "Darwin" ]; then
    DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:../../../lib ./${EXE_NAME} --input "This movie is so amazing" 2&> ${EXE_NAME}.log
else
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../../../lib ./${EXE_NAME} --input "This movie is so amazing" 2&> ${EXE_NAME}.log
fi
result=`grep "The sentiment score between 0 and 1.*\=" sentiment_analysis_rnn.log | cut -d '=' -f2`
if [ $result ==  "0.986632" ];
then
    echo "PASS: ${EXE_NAME} correctly predicted the sentiment with score = $result"
    exit 0
else
    echo "FAIL: ${EXE_NAME} FAILED to predict the sentiment with score = $result"
    exit 1
fi