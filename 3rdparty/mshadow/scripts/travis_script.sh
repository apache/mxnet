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

# main script of travis
if [ ${TASK} == "lint" ]; then
    python3 dmlc-core/scripts/lint.py mshadow all mshadow mshadow-ps || exit -1
fi

if [ ${TASK} == "doc" ]; then
    doxygen doc/Doxyfile 2>log.txt
    (cat log.txt| grep -v ENABLE_PREPROCESSING |grep -v "unsupported tag" |grep nothing) && exit -1
fi

if [ ${TASK} == "build" ]; then
    cd guide
    echo "USE_BLAS=blas" >> config.mk
    make all || exit -1
    cd mshadow-ps
    echo "USE_BLAS=blas" >> config.mk
    echo "USE_RABIT_PS=0" >> config.mk    
    make local_sum.cpu || exit -1
fi
