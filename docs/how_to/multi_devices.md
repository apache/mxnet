# Run MXNet on Multiple CPU/GPUs with Data Parallel

MXNet supports trainig with multiple CPUs and GPUs since the very
beginning. Almost any program using MXNet's provided training modules, such as
[python/mxnet.model](https://github.com/dmlc/mxnet/blob/master/python/mxnet/model.py),
can be efficiently run over multiple devices.

## Data Parallelism

In default MXNet uses data parallelism to partition the workload over multiple
devices. Assume there are *n* devices, then each one will get the complete model
and train it on *1/n* of the data. The results such as the gradient and
updated model are communicated cross these devices.

## Multiple GPUs within a Single Machine

### Workload Partitioning

If using data parallelism, MXNet will evenly partition a minbatch in each
GPUs. Assume we train with batch size *b* and *k* GPUs, then in one iteration
each GPU will perform forward and backward on a batch with size *b/k*. The
gradients are then summed over all GPUs before updating the model.

In ideal case, *k* GPUs will provide *k* time speedup comparing to the single
GPU. In addition, assume the model has size *m* and the temporal workspace is
*t*, then the memory footprint of each GPU will be *m+t/k*. In other words, we
can use a large batch size for multiple GPUs.

### How to Use

> To use GPUs, we need to compiled MXNet with GPU support. For
> example, set `USE_CUDA=1` in `config.mk` before `make`. (see
> [MXNet installation guide](build.html) for more options).

If a machine has one or more than one GPU cards installed, then each card is
labeled by a number starting from 0. To use a particular GPU, one can often
either specify the context `ctx` in codes or pass `--gpus` in commandlines. For
example, to use GPU 0 and 2 in python one can often create a model with
```python
import mxnet as mx
model = mx.model.FeedForward(ctx=[mx.gpu(0), mx.gpu(2)], ...)
```
while if the program accepts a `--gpus` flag such as
[example/image-classification](https://github.com/dmlc/mxnet/tree/master/example/image-classification),
then we can try
```bash
python train_mnist.py --gpus 0,2 ...
```

### Advanced Usage

If the GPUs are have different computation power, we can partition the workload
according to their powers. For example, if GPU 0 is 3 times faster than GPU 2,
then we provide an additional workload option `work_load_list=[3, 1]`, see
[model.fit](../packages/python/model.html#mxnet.model.FeedForward.fit) for more
details.

Training with multiple GPUs should have the same results as a single GPU if all
other hyper-parameters are the same. But in practice the results vary mainly due
to the randomness of I/O (random order or other augmentations), weight
initialization with different seeds, and CUDNN.

We can control where the gradient is aggregated and model updating if performed
by creating different `kvstore`, which is the module for data
communication. There are three options,
which vary on speed and memory consumption:

```eval_rst
==========================  ====================  ================
kvstore type                gradient aggregation  weight updating
==========================  ====================  ================
``local_update_cpu``        CPU                   CPU
``local_allreduce_cpu``     CPU                   all GPUs
``local_allreduce_device``  one GPU               all GPUs
==========================  ====================  ================
```

Here
- `local_update_cpu`: gradients are first copied to CPU memory, and aggregated
  on CPU. Then we update the weight on CPU and copy back the updated weight to
  GPUs. It is suitable when the layer model size is not large, such as
  convolution layers.

- `local_allreduce_cpu` is similar to `local_update_cpu` except that the
  aggregated gradients are copied back to each GPUs, and the weight is updated
  there. Note that, comparing to `local_update_cpu`, each weight is updated by
  *k* times if there are *k* GPUs. But it might be still faster when the model
  size is large, such as fully connected layers, in which GPUs is much faster
  than CPU. Also note that, it may use more GPU memory because we need to store
  the variables needed by the updater in GPU memory.

- `local_allreduce_device`, or simplified as `device`, is similar to
   `local_allreduce_cpu` except that the we use a particular GPU to aggregated
   the gradients. It may be faster than `local_allreduce_cpu` if the gradient
   size is huge, where the gradient summation operation could be the
   bottleneck. However, it uses even more GPU memory since we need to store the
   aggregated gradient on GPU.

The `kvstore` type is `local` in default. It will choose `local_update_cpu` if the
weight size of each layer is less than 1Mb, which can be changed by
the environment varialbe `MXNET_KVSTORE_BIGARRAY_BOUND`, and
`local_allreduce_cpu` otherwise.

## Distributed Training with Multiple Machines

### Data Consistency Model

MXNet provides two `kvstore` types with different trade-off between convergence
and speed when using multiple machines.

- `dist_sync` behaviors similarly to `local_update_cpu`, where the gradients are
  first aggregated before updating the weight. But a difference is that
  `batch-size` now means the batch size used on each machine. So if there are *n*
  machines and we use batch size *b*, then `dist_sync` will give the same
  results for using batch size *n\*b* on a single machine.

- `dist_async` remove the aggregation operation in `dist_sync`. The weight is
  updated once received gradient from any machine. The updating is atomic,
  namely no two updatings happen on the same weight at the same time. However,
  the order is not guaranteed.

Roughly speaking, `dist_sync` runs slower than `dist_async` due the extra
aggregation, but it provides deterministic results. We suggest to use
`dist_sync` if the speed is not significantly slower than `dist_async`. Namely,
keep all hyper-parameters fixed, changes `kvstore` from `dist_sync` to
`dist_async`, if the former is not much slower than the latter, then we suggest
to use the former. Please refer to ps-lite's
[document](http://ps-lite.readthedocs.org/en/latest/overview.html) to see more
information about these two data consistency models.

### How to Launch a Job

> To use distributed training, we need to compile with `USE_DIST_KVSTORE=1`
> (see [MXNet installation guide](build.html) for more options).

Launching a distributed job is little bit different than running on a single
machine. MXNet provides
[tools/launch.py](https://github.com/dmlc/mxnet/blob/master/tools/launch.py) to
start a job by using `ssh`, `mpi`, `sge`, or `yarn`.

Assume we are at the directory `mxnet/example/image-classification`.  and want
to train mnist with lenet by using
[train_mnist.py](https://github.com/dmlc/mxnet/blob/master/example/image-classification/train_mnist.py).
On a single machine  we can run by

```bash
python train_mnist.py --network lenet
```

Now if there are two ssh-able machines, and we want to train it on these two
machines.
First we save the IPs (or hostname) of these two machines in file `hosts`, e.g.

```bash
$ cat hosts
172.30.0.172
172.30.0.171
```

Next if the mxnet folder is accessible by both machines, e.g. on a
[network filesystem](https://help.ubuntu.com/lts/serverguide/network-file-system.html),
then we can run by

```bash
../../tools/launch.py -n 2 --launcher ssh -H hosts python train_mnist.py --network lenet --kv-store dist_sync
```

Note that, besides the single machine arguments, here we

- use `launch.py` to submit the job
- provide launcher, `ssh` if all machines are ssh-able, `mpi` if `mpirun` is
  available, `sge` for Sun Grid Engine, and `yarn` for Apache Yarn.
- `-n` number of worker nodes to run
- `-H` the host file which is required by `ssh` and `mpi`
- `--kv-store` use either `dist_sync` or `dist_async`


### Synchronize Directory

Now consider if the mxnet folder is not accessible. We can first copy the MXNet
library to this folder by
```bash
cp -r ../../python/mxnet .
cp -r ../../lib/libmxnet.so mxnet
```

then ask `launch.py` to synchronize the current directory to all machines'
 `/tmp/mxnet` directory with `--sync-dst-dir`

```bash
../../tools/launch.py -n 2 -H hosts --sync-dst-dir /tmp/mxnet \
   python train_mnist.py --network lenet --kv-store dist_sync
```

### Use a Particular Network Interface

MXNet often chooses the first available network interface. But for machines have
multiple interface, we can specify which network interface to use for data
communication by the environment variable `DMLC_INTERFACE`. For example, to use
the interface `eth0`, we can

```
export DMLC_INTERFACE=eth0; ../../tools/launch.py ...
```

### Debug Connection

Set`PS_VERBOSE=1` to see the debug logging, e.g
```
export PS_VERBOSE=1; ../../tools/launch.py ...
```

### More

- See more launch options by `../../tools/launch.py -h`
- See more options of [ps-lite](http://ps-lite.readthedocs.org/en/latest/how_to.html)
