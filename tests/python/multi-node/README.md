# Tests for multi-devices and multi-machines

## How to use

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

# Results

## Datasets and machines

| name | class | image size | training | testing |
| ---- | ----: | ---------: | -------: | ------: |
| CIFAR10 | 10 | 28 × 28 × 3 | 60,000  | 10,000 |
| ILSVRC12 | 1,000 | 227 × 227 × 3 | 1,281,167 | 50,000 |

| name | hardware | software |
| --- | --- | --- |
| gtx980_cluster | dual Xeon E5-2680 v2, dual GTX 980, 1G ethernet | GCC 4.8, CUDA 7.5, CUDNN v3 |


## Inception on ILSVRC12 and gtx980_cluster

It uses [dist_imagenet_inception.py](dist_imagenet_inception.py)

### Single GTX 980

| param | value |
| --- | --- |
| batch size | 48 |
| learning rate | 0.05 |
| momentum      | 0.9 |
| wd            | 0.00001 |

| epoch | train accuracy | valid accuracy | time |
| ---  | --- | --- | --- |
| 10 | 0.596858 | 0.568018 | 22800 |
| 20 | 0.634014 | 0.588112 | 22820 |
