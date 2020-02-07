#!/bin/bash
# export NVIDIA_VISIBLE_DEVICES=0; 
# export DMLC_NUM_WORKER=1; 
# export DMLC_WORKER_ID=0; 
# export DMLC_ROLE=worker; 
# export BYTEPS_LOG_LEVEL=DEBUG; 
# python3 cck-test.py

# export BYTEPS_ENABLE_GDB=1
# export BYTEPS_LOG_LEVEL=INFO
# export PS_VERBOSE=2
# export BYTEPS_TRACE_ON=1
# export BYTEPS_TRACE_END_STEP=20
# export BYTEPS_TRACE_START_STEP=1
# export BYTEPS_TRACE_DIR="/home/ubuntu/byteps_traces"

export NVIDIA_VISIBLE_DEVICES=0,1
python3 ./tools/launch.py -n 1 -s 1 -H ./hostfile --byteps-launch \
    "python3 ./tests/nightly/dist_device_sync_kvstore_byteps.py"