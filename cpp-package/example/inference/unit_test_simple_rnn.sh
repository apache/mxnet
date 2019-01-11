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



# Running the example with dog image.
if [ "$(uname)" == "Darwin" ]; then
    DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:../../../lib ./simple_rnn  2&> simple_rnn.log
else
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../../../lib ./simple_rnn  2&> simple_rnn.log
fi
predicted_line=`grep -o '\[[^]]*\]$' simple_rnn.log`
num_words=`echo $predicted_line | cut -d '[' -f 2 | cut -d ']' -f 1 | awk -F ' ' '{print NF}'`
if [ $num_words == 35 ];
then
    echo "PASS: simple_rnn correctly predicted following sequence of $num_words words."
    echo "Sequence: $predicted_line"
    exit 0
else
    echo "FAIL: simple_rnn FAILED to predict the output sequence."
    exit 1
fi