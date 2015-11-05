# Tests for multi-devices and multi-machines

- `local_*` for multi-devices and single machine. Requires at least two GPUs.
- `dist_sync_*` for multi-machines with BSP synchronizations
- `dist_async_*` for multi-machines with asynchronous SGD

(Note that `CUDNN` leads to randomness, need to disable if comparing to the baseline)

- runs on local machine with two servers and two workers

```
../../../ps-lite/tracker/dmlc_local.py -n 2 -s 2 python dist_sync_mlp.py
```

- runs on multiple machines with machine names in `hosts`

```
../../../ps-lite/tracker/dmlc_mpi.py -n 2 -s 2 -H hosts python dist_sync_mlp.py
```

See more examples on [example/distributed-training/](../../../example/distributed-training)
