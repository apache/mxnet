export PYTHONPATH=./python/
export MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
export MXNET_SUBGRAPH_VERBOSE=0
export DMLC_LOG_STACK_TRACE_DEPTH=10

test_args=(
    "-n 4 --launcher local python3 dist_device_sync_kvstore.py"
    "-n 4 --launcher local python3 dist_device_sync_kvstore_custom.py"
    "--p3 -n 4 --launcher local python3 dist_device_sync_kvstore_custom.py"
    "-n 4 --launcher local python3 dist_sync_kvstore.py --type=init_gpu" 
)

for arg in "${test_args[@]}"; do
    echo "$i"
    python3 ../../tools/launch.py "$arg"
    if [ $? -ne 0 ]; then
        return $?
    fi 
done