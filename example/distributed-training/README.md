# Examples for Distributed Training

## How to use

- runs on multiple machines with machine names in `hosts`

```
../../../ps-lite/tracker/dmlc_mpi.py -n 2 -s 2 -H hosts python dist_sync_mlp.py
```

# Results

Experimental results. The full training logs are available at [log/](log/). The
results tables are generated from [tools/parse_log.py](../../tools/parse_log.py).

## Machines

| name | hardware | software |
| --- | --- | --- |
| GTX980-cluster | dual Xeon E5-2680 v2, dual GTX 980, 1G Ethernet | GCC 4.8, CUDA 7.5, CUDNN v3 |
| EC2-g2.8-cluster | Xeon E5-2670, dual GRID K520, 10G Ethernet | GCC 4.8, CUDA 7.5, CUDNN v3 |

## Datasets

| name | class | image size | training | testing |
| ---- | ----: | ---------: | -------: | ------: |
| CIFAR10 | 10 | 28 × 28 × 3 | 60,000  | 10,000 |
| ILSVRC12 | 1,000 | 227 × 227 × 3 | 1,281,167 | 50,000 |

## Incepption on CIFAR10 using GTX980-cluster

Based on [train_cifar10.py](train_cifar10.py)

### Single GTX 980

- `batch_size = 256`, `learning_rate = .1`

| epoch | train accuracy | valid accuracy | time |
| ---  | --- | --- | --- |
| 10 | 0.904931 | 0.867488 | 71.1 |
| 20 | 0.953142 | 0.873598 | 70.9 |

full log [log/cifar10/incept_2](log/cifar10/incept_2)

- `batch_size = 128`, `learning_rate = .1`

| epoch | train accuracy | valid accuracy | time |
| ---  | --- | --- | --- |
| 10 | 0.905315 | 0.867288 | 82.3 |
| 20 | 0.954310 | 0.889022 | 82.8 |
| 30 | 0.973641 | 0.898237 | 82.6 |
| 40 | 0.982739 | 0.899940 | 82.9 |

full log [log/cifar10/incept_3](log/cifar10/incept_3)

### 5 machines with 10 GTX 980, BSP

- `batch_size = 512`, `learning_rate = .1`

| epoch | train accuracy | valid accuracy | time |
| ---  | --- | --- | --- |
| 10 | 0.828193 | 0.804980 | 6.9 |
| 20 | 0.901002 | 0.831250 | 6.9 |
| 30 | 0.938349 | 0.860444 | 7.0 |
| 40 | 0.953838 | 0.859629 | 6.9 |

full log [log/cifar10/incept_5](log/cifar10/incept_5)

- `batch_size = 512`, `learning_rate = .4`

| epoch | train accuracy | valid accuracy | time |
| ---  | --- | --- | --- |
| 10 | 0.799032 | 0.763262 | 6.9 |
| 20 | 0.903737 | 0.847594 | 7.0 |
| 30 | 0.945143 | 0.854461 | 7.0 |
| 40 | 0.962789 | 0.869531 | 7.0 |

full log [log/cifar10/incept_6](log/cifar10/incept_6)

- `batch_size = 256`, `learning_rate = .1`

| epoch | train accuracy | valid accuracy | time |
| ---  | --- | --- | --- |
| 10 | 0.863264 | 0.824479 | 11.1 |
| 20 | 0.923930 | 0.858333 | 11.5 |
| 30 | 0.951104 | 0.862540 | 11.2 |
| 40 | 0.968971 | 0.873658 | 11.0 |

full log [log/cifar10/incept_7](log/cifar10/incept_7)

- `batch_size = 256`, `learning_rate = .4`

| epoch | train accuracy | valid accuracy | time |
| ---  | --- | --- | --- |
| 10 | 0.867188 | 0.837039 | 10.7 |
| 20 | 0.931913 | 0.869331 | 10.7 |
| 30 | 0.959867 | 0.874098 | 10.7 |
| 40 | 0.975832 | 0.888101 | 10.7 |

full log [log/cifar10/incept_4](log/cifar10/incept_4)

### 5 machines with 10 GTX 980, Async

## Inception on ILSVRC12 using GTX980-cluster

Based on [train_imagenet.py](train_imagenet.py)

### Single GTX 980

- `batch_size = 48`, `learning_rate = 0.05`

| epoch | train accuracy | valid accuracy | time |
| ---  | --- | --- | --- |
| 5 | 0.532555 | 0.526332 | 22783.1 |
| 10 | 0.596858 | 0.568018 | 22800.3 |
| 15 | 0.621247 | 0.578255 | 22854.4 |
| 20 | 0.634014 | 0.588112 | 22820.8 |

full log [log/ilsvrc12/incept_1](log/ilsvrc12/incept_1)


- `batch_size = 48`, `learning_rate = 0.05`

| epoch | train accuracy | valid accuracy | time |
| --- | --- | --- | --- |
| 5 | 0.517945 | 0.509897 | 16026.8 |
| 10 | 0.579599 | 0.548764 | 16291.1 |

full log [log/ilsvrc12/incept_6](log/ilsvrc12/incept_6)

running

### 5 machines with 10 GTX 980, BSP

- `batch_size = 96`, `learning_rate = 0.05`

| epoch | train accuracy | valid accuracy | time |
| ---  | --- | --- | --- |
| 5 | 0.547913 | 0.535645 | 3013.3 |
| 10 | 0.647100 | 0.600572 | 3040.4 |
| 15 | 0.693372 | 0.621829 | 3039.6 |
| 19 | 0.717419 | 0.631606 | 3042.3 |

full log [log/ilsvrc12/incept_2](log/ilsvrc12/incept_2)

- `batch_size = 96`, `learning_rate = 0.1`

| epoch | train accuracy | valid accuracy | time |
| --- | --- | --- | --- |
| 5 | 0.548047 | 0.531654 | 2933.4 |
| 10 | 0.646048 | 0.589231 | 2939.0 |
| 15 | 0.691162 | 0.609613 | 2936.2 |
| 20 | 0.718662 | 0.615359 | 2936.1 |

full log [log/ilsvrc12/incept_5](log/ilsvrc12/incept_5)

### 5 machines with 10 GTX 980, Async

- `batch_size = 96`, `learning_rate = 0.05`

| epoch | train accuracy | valid accuracy | time |
| --- | --- | --- | --- |
| 5 | 0.528214 | 0.502539 | 2725.1 |
| 10 | 0.623501 | 0.569418 | 2734.7 |
| 15 | 0.660391 | 0.597289 | 2735.3 |
| 20 | 0.714859 | 0.605626 | 2172.0 |

full log [log/ilsvrc12/incept_4](log/ilsvrc12/incept_4)

- `batch_size = 96`, `learning_rate = 0.1`

| epoch | train accuracy | valid accuracy | time |
| ---  | --- | --- | --- |
| 5 | 0.518088 | 0.505218 | 2718.2 |
| 10 | 0.602544 | 0.565539 | 2714.5 |
| 15 | 0.632609 | 0.584213 | 2720.2 |
| 20 | 0.669050 | 0.595749 | 2128.7 |

full log [log/ilsvrc12/incept_3](log/ilsvrc12/incept_3)
