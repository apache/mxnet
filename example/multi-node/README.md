# Examples for Distributed Training

## How to use


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
| GTX980-cluster | dual Xeon E5-2680 v2, dual GTX 980, 1G Ethernet | GCC 4.8, CUDA 7.5, CUDNN v3 |
| EC2-g2.8-cluster | Xeon E5-2670, dual GRID K520, 10G Ethernet | GCC 4.8, CUDA 7.5, CUDNN v3 |

## Incepption on CIFAR10 using GTX980-cluster

Based on [train_cifar10.py](train_cifar10.py)

## Inception on ILSVRC12 using GTX980-cluster

Based on [train_imagenet.py](train_imagenet.py)

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

full log [log/ilsvrc12/incept_1](log/ilsvrc12/incept_1)

### 5 machines with 10 GTX 980, BSP

| param | value |
| --- | --- |
| batch size | 96 |
| learning rate | 0.05 |
| momentum      | 0.9 |
| wd            | 0.00001 |

| epoch | train accuracy | valid accuracy | time |
| ---  | --- | --- | --- |
| 10 | 0.647994 | 0.602307 | 3042 |
| 19 | 0.719019 | 0.631338 | 3042 |

full log [log/ilsvrc12/incept_2](log/ilsvrc12/incept_2)

### 5 machines with 10 GTX 980, Async

| param | value |
| --- | --- |
| batch size | 96 |
| learning rate | 0.1 |
| momentum      | 0.9 |
| wd            | 0.00001 |

| epoch | train accuracy | valid accuracy | time |
| ---  | --- | --- | --- |
| 10 | 0.602544 | 0.565539 | 2730 |
| 20 | 0.669050 | 0.589471 | 1838 |

full log [log/ilsvrc12/incept_3](log/ilsvrc12/incept_3)
