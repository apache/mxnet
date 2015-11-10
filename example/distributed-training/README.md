# Examples for Distributed Training

## How to use

- If machines are ssh-able. First prepare a
  file with all hostnames, such as `cat hosts`

  ```bash
  172.30.0.172
  172.30.0.171
  ```

  Next prepare a working directory, and then copy mxnet libraries and the
  trainig codes

  ```bash
  cp -r ../../python/mxnet working_dir
  cp -r ../../lib/libmxnet.so working_dir/mxnet
  cp -r *.py working_dir
  ```

  Then start the jobs with 2 workers (with 2 servers):

  ```bash
  cd workding_dir
  mxnet_dir/tracker/dmlc_ssh.py -n 2 -s 2 -H hosts python dist_sync_mlp.py
  ```

- If mxnet is on a shared filesystem and `mpirun` is availabe,

  ```
  ../../tracker/dmlc_mpi.py -n 2 -s 2 -H hosts python dist_sync_mlp.py
  ```

- We can also submit the jobs by resource managers such as `Yarn`

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

### System Performance

| cluster | # machines | # GPUs | batch size / GPU | kvstore | epoch time (sec) |
| --- | --- | --- | --- |
| GTX980 | 1 | 1 |  256 | `local` | 71 |
|  - | 1 | 1 | 128 | `dist_sync` | 128 |
| - | 5 | 10 | 256 | `dist_sync` | 7 |
| - | 5 | 10 | 128 | `dist_sync` | 11 |


### Single GTX 980

- `batch_size = 256`, `learning_rate = .1`

| epoch | train accuracy | valid accuracy |
| ---  | --- | --- |
| 10 | 0.904931 | 0.867488 |
| 20 | 0.953142 | 0.873598 |

full log [log/cifar10/incept_2](log/cifar10/incept_2)

- `batch_size = 128`, `learning_rate = .1`

| epoch | train accuracy | valid accuracy |
| ---  | --- | --- |
| 10 | 0.905315 | 0.867288 |
| 20 | 0.954310 | 0.889022 |
| 30 | 0.973641 | 0.898237 |
| 40 | 0.982739 | 0.899940 |

full log [log/cifar10/incept_3](log/cifar10/incept_3)

### 5 machines with 10 GTX 980, BSP

- `batch_size = 512`, `learning_rate = .1`

| epoch | train accuracy | valid accuracy |
| ---  | --- | --- |
| 10 | 0.828193 | 0.804980 |
| 20 | 0.901002 | 0.831250 |
| 30 | 0.938349 | 0.860444 |
| 40 | 0.953838 | 0.859629 |

full log [log/cifar10/incept_5](log/cifar10/incept_5)

- `batch_size = 512`, `learning_rate = .4`

| epoch | train accuracy | valid accuracy |
| ---  | --- | --- |
| 10 | 0.799032 | 0.763262 |
| 20 | 0.903737 | 0.847594 |
| 30 | 0.945143 | 0.854461 |
| 40 | 0.962789 | 0.869531 |

full log [log/cifar10/incept_6](log/cifar10/incept_6)

- `batch_size = 256`, `learning_rate = .1`

| epoch | train accuracy | valid accuracy |
| ---  | --- | --- |
| 10 | 0.863264 | 0.824479 |
| 20 | 0.923930 | 0.858333 |
| 30 | 0.951104 | 0.862540 |
| 40 | 0.968971 | 0.873658 |

full log [log/cifar10/incept_7](log/cifar10/incept_7)

- `batch_size = 256`, `learning_rate = .4`

| epoch | train accuracy | valid accuracy |
| ---  | --- | --- |
| 10 | 0.867188 | 0.837039 |
| 20 | 0.931913 | 0.869331 |
| 30 | 0.959867 | 0.874098 |
| 40 | 0.975832 | 0.888101 |

full log [log/cifar10/incept_4](log/cifar10/incept_4)

### 5 machines with 10 GTX 980, Async

## Inception on ILSVRC12 using GTX980-cluster

Based on [train_imagenet.py](train_imagenet.py)

### System Performance

| cluster | # machines | # GPUs | batch size / GPU | kvstore | epoch time (sec) |
| --- | --- | --- | --- |
| GTX980 | 1 | 1 |  48 | `local` | ? |
| GTX980 | 1 | 2 |  48 | `local` | ? |
| - | 5 | 10 |  48 | `dist_sync` | 3000 |
| - | 5 | 10 |  48 | `dist_async` | 2800 |
| EC2-g2.8 | 1 | 4 | 36 |  `local` | 14203 |
| - | 10 | 40 | 36 |  `dist_sync` | 1422 |


### Single GTX 980

- `batch_size = 48`, `learning_rate = 0.05`

| epoch | train accuracy | valid accuracy |
| ---  | --- | --- |
| 5 | 0.532555 | 0.526332 |
| 10 | 0.596858 | 0.568018 |
| 15 | 0.621247 | 0.578255 |
| 20 | 0.634014 | 0.588112 |

full log [log/ilsvrc12/incept_1](log/ilsvrc12/incept_1)


- `batch_size = 48`, `learning_rate = 0.1`

| epoch | train accuracy | valid accuracy |
| --- | --- | --- |
| 5 | 0.517945 | 0.509897 |
| 10 | 0.579599 | 0.548764 |
| 15 | 0.604185 | 0.570897 |

full log [log/ilsvrc12/incept_6](log/ilsvrc12/incept_6)

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

| epoch | train accuracy | valid accuracy |
| --- | --- | --- |
| 5 | 0.548047 | 0.531654 |
| 10 | 0.646048 | 0.589231 |
| 15 | 0.691162 | 0.609613 |
| 20 | 0.718662 | 0.615359 |

full log [log/ilsvrc12/incept_5](log/ilsvrc12/incept_5)

### 5 machines with 10 GTX 980, Async

- `batch_size = 96`, `learning_rate = 0.05`

| epoch | train accuracy | valid accuracy |
| --- | --- | --- |
| 5 | 0.528214 | 0.502539 |
| 10 | 0.623501 | 0.569418 |
| 15 | 0.660391 | 0.597289 |
| 20 | 0.714859 | 0.605626 |

full log [log/ilsvrc12/incept_4](log/ilsvrc12/incept_4)

- `batch_size = 96`, `learning_rate = 0.1`

| epoch | train accuracy | valid accuracy |
| ---  | --- | --- |
| 5 | 0.518088 | 0.505218 |
| 10 | 0.602544 | 0.565539 |
| 15 | 0.632609 | 0.584213 |
| 20 | 0.669050 | 0.595749 |

full log

### 10 EC2 g2.8x instances, Sync

- `batch_size = 36 * 4`, `learning_rate = 0.05`

| epoch | train accuracy | valid accuracy |
| --- | --- | --- |
| 5 | 0.516417 | 0.506337 |

full log [log/ilsvrc12/incept_7](log/ilsvrc12/incept_7)
