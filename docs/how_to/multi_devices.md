# Run MXNet on Multiple CPU/GPUs with Data Parallelism

_MXNet_ supports training with multiple CPUs and GPUs, which may be located on different physical machines.

## Data Parallelism vs Model Parallelism

By default, _MXNet_ uses data parallelism to partition the workload over multiple
devices.
Assume there are *n* devices.
Then each one will receive a copy of the complete model
and train it on *1/n* of the data.
The results such as the gradient and
updated model are communicated across these devices.

MXNet also supports model parallelism.
In this approach, each device holds onto only part of the model.
This proves useful when the model is too large to fit onto a single device.
As an example, see the following [ tutorial](./model_parallel_lstm.md)
which shows how to use model parallelism for training a multi-layer LSTM model.
For the duration of this tutorial, we'll focus on data parallelism.  

## Multiple GPUs within a Single Machine

### Workload Partitioning

By default, _MXNet_ partitions a data batch evenly among the available GPUs.
Assume batch size *b* and *k* GPUs, then in one iteration
each GPU will perform forward and backward on *b/k* examples. The
gradients are then summed over all GPUs before updating the model.

### How to Use

> To use GPUs, we need to compiled MXNet with GPU support. For
> example, set `USE_CUDA=1` in `config.mk` before `make`. (see
> [MXNet installation guide](http://mxnet.io/get_started/setup.html) for more options).

If a machine has one or more than one GPU cards installed,
then each card is labeled by a number starting from 0.
To use a particular GPU, one can often either
specify the context `ctx` in codes
or pass `--gpus` in the command line.
For example, to use GPU 0 and 2 in python,
one can typically create a model with
```python
import mxnet as mx
model = mx.model.FeedForward(ctx=[mx.gpu(0), mx.gpu(2)], ...)
```
while if the program accepts a `--gpus` flag (as seen in
[example/image-classification](https://github.com/dmlc/mxnet/tree/master/example/image-classification)),
then we can try
```bash
python train_mnist.py --gpus 0,2 ...
```

### Advanced Usage
If the available GPUs are not equally powerful,
we can partition the workload accordingly.
For example, if GPU 0 is 3 times faster than GPU 2,
then we might use the workload option `work_load_list=[3, 1]`,
see [model.fit](../api/python/model.html#mxnet.model.FeedForward.fit)
for more details.

Training with multiple GPUs should yield the same results
as training on a single GPU if all other hyper-parameters are the same.
In practice, the results may exhibit small differences,
owing to the randomness of I/O (random order or other augmentations),
weight initialization with different seeds, and CUDNN.

We can control on which devices the gradient is aggregated
and on which device the model is updated via [`KVStore`](http://mxnet.io/api/python/kvstore.html),
the _MXNet_ module that supports data communication.
One can either use `mx.kvstore.create(type)` to get an instance
or use the program flag `--kv-store type`.

There are two commonly used types,

- `local`: all gradients are copied to CPU memory and weights are updated there.
- `device`: both gradient aggregation and weight updates are run on GPUs.
With this setting, the `KVStore` also attempts to use GPU peer-to-peer communication,
potentially accelerating the communication.
Note that this option may result in higher GPU memory usage.

When using a large number of GPUs, e.g. >=4, we suggest using `device` for better performance.

## Distributed Training with Multiple Machines

`KVStore` also supports a number of types for running on multiple machines.

- `dist_sync` behaves similarly to `local` but exhibits one major difference.
  With `dist_sync`, `batch-size` now means the batch size used on each machine.
  So if there are *n* machines and we use batch size *b*,
  then `dist_sync` behaves like `local` with batch size *n\*b*.
- `dist_device_sync` is identical to `dist_sync`  with the difference similar to `device` vs `local`.  
- `dist_async`  performs asynchronous updates.
  The weight is updated whenever gradients are received from any machine.
  The update is atomic, i.e., no two updates happen on the same weight at the same time.
  However, the order is not guaranteed.

### How to Launch a Job

> To use distributed training, we need to compile with `USE_DIST_KVSTORE=1`
> (see [MXNet installation guide](http://mxnet.io/get_started/setup.html) for more options).

Launching a distributed job is a bit different from running on a single
machine. MXNet provides
[tools/launch.py](https://github.com/dmlc/mxnet/blob/master/tools/launch.py) to
start a job by using `ssh`, `mpi`, `sge`, or `yarn`.

Assume we are at the directory `mxnet/example/image-classification`
and want to train lenet to classify MNIST images, as demonstrated here:
[train_mnist.py](https://github.com/dmlc/mxnet/blob/master/example/image-classification/train_mnist.py).

On a single machine, we can run:

```bash
python train_mnist.py --network lenet
```

Now, say we are given two ssh-able machines and we want to train lenet on these two
machines.
First, we save the IPs (or hostname) of these two machines in file `hosts`, e.g.

```bash
$ cat hosts
172.30.0.172
172.30.0.171
```

Next, if the mxnet folder is accessible by both machines, e.g. on a
[network filesystem](https://help.ubuntu.com/lts/serverguide/network-file-system.html),
then we can run by

```bash
python ../../tools/launch.py -n 2 --launcher ssh -H hosts python train_mnist.py --network lenet --kv-store dist_sync
```

Note that here we

- use `launch.py` to submit the job
- provide launcher, `ssh` if all machines are ssh-able, `mpi` if `mpirun` is
  available, `sge` for Sun Grid Engine, and `yarn` for Apache Yarn.
- `-n` number of worker nodes to run
- `-H` the host file which is required by `ssh` and `mpi`
- `--kv-store` use either `dist_sync` or `dist_async`


### Synchronize Directory

Now consider if the mxnet folder is not accessible.
We can first copy the `MXNet` library to this folder by
```bash
cp -r ../../python/mxnet .
cp -r ../../lib/libmxnet.so mxnet
```

then ask `launch.py` to synchronize the current directory to all machines'
 `/tmp/mxnet` directory with `--sync-dst-dir`

```bash
python ../../tools/launch.py -n 2 -H hosts --sync-dst-dir /tmp/mxnet \
   python train_mnist.py --network lenet --kv-store dist_sync
```

### Use a Particular Network Interface

_MXNet_ often chooses the first available network interface.
But for machines that have multiple interfaces,
we can specify which network interface to use for data
communication by the environment variable `DMLC_INTERFACE`.
For example, to use the interface `eth0`, we can

```
export DMLC_INTERFACE=eth0; python ../../tools/launch.py ...
```

### Debug Connection

Set`PS_VERBOSE=1` to see the debug logging, e.g
```
export PS_VERBOSE=1; python ../../tools/launch.py ...
```

### More

- See more launch options by `python ../../tools/launch.py -h`
- See more options of [ps-lite](http://ps-lite.readthedocs.org/en/latest/how_to.html)
