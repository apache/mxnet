#!/bin/bash
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
