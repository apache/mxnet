# Test multi-devices and multi-machines

must disable `CUDNN`

`local_*` for multi-devices and single machine. Requires two GPUs.


`dist_*` for multi-machines. Run in local machine with 2 workers (requires at
least two gpus) and 2 servers.


```
ln -s ../../../dmlc-core/tracker/dmlc_local.py .
./dmlc_local.py -n 2 -s 2 ./dist_sync_mlp.py
```
