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

export PYTHONPATH=./python/
export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
export MXNET_SUBGRAPH_VERBOSE=0
export DMLC_LOG_STACK_TRACE_DEPTH=10

test_kvstore() {
    test_args=(
        "-n 4 --launcher local python3 dist_device_sync_kvstore.py"
        "-n 4 --launcher local python3 dist_device_sync_kvstore_custom.py"
        "--p3 -n 4 --launcher local python3 dist_device_sync_kvstore_custom.py"
        "-n 4 --launcher local python3 dist_sync_kvstore.py --type=init_gpu"
    )

    for arg in "${test_args[@]}"; do
        python3 ../../tools/launch.py $arg
        if [ $? -ne 0 ]; then
            return $?
        fi
    done
}

test_horovod() {
    echo "localhost slots=2" > hosts
    mpirun -np 2 --hostfile hosts --bind-to none --map-by slot -mca pml ob1 \
        -mca btl ^openib python3 dist_device_sync_kvstore_horovod.py
    if [ $? -ne 0 ]; then
        return $?
    fi
}

test_kvstore
test_horovod

exit $errors
