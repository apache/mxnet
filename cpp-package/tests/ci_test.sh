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
cd $(dirname $(readlink -f $0))/../example
echo $PWD
export LD_LIBRARY_PATH=$(readlink -f ../../lib):$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
ls -l ../../lib/

cp ../../build/cpp-package/example/test_score .
./get_mnist.sh
./test_score 0.93
