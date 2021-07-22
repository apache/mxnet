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

# set -x
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../third_party/lib
if [ $# -lt 3 ]; then
    echo "usage: $0 num_servers num_workers bin [args..]"
    exit -1;
fi

num_servers=$1
shift
num_workers=$1
shift
bin=$1
shift
arg="-num_servers ${num_servers} -num_workers ${num_workers} -log_dir log $@"


# killall -q $(basename ${bin})
# killall -q ${bin}

# start the scheduler
Sch="role:SCHEDULER,hostname:'127.0.0.1',port:8001,id:'H'"
${bin} -my_node ${Sch} -scheduler ${Sch} ${arg} &

# start servers
for ((i=0; i<${num_servers}; ++i)); do
    port=$((9600 + ${i}))
    N="role:SERVER,hostname:'127.0.0.1',port:${port},id:'S${i}'"
    ${bin} -my_node ${N} -scheduler ${Sch} ${arg} &
done

# start workers
for ((i=0; i<${num_workers}; ++i)); do
    port=$((9500 + ${i}))
    N="role:WORKER,hostname:'127.0.0.1',port:${port},id:'W${i}'"
    ${bin} -my_node ${N} -scheduler ${Sch} ${arg} &
done

wait
