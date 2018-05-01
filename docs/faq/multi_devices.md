# Run MXNet on Multiple CPU/GPUs with Data Parallelism

_MXNet_ supports training with multiple CPUs and GPUs, which may be located on different physical machines.

## Data Parallelism vs Model Parallelism

By default, _MXNet_ uses data parallelism to partition the workload over multiple
devices.
Assume there are *n* devices.
Then each one will receive a copy of the complete model
and train it on *1/n* of the data.
The results such as gradients and
updated model are communicated across these devices.

MXNet also supports model parallelism.
In this approach, each device holds onto only part of the model.
This proves useful when the model is too large to fit onto a single device.
As an example, see the following [tutorial](./model_parallel_lstm.md)
which shows how to use model parallelism for training a multi-layer LSTM model.
In this tutorial, we'll focus on data parallelism.

## Multiple GPUs within a Single Machine

### Workload Partitioning

By default, _MXNet_ partitions a data batch evenly among the available GPUs.
Assume a batch size *b* and assume there are *k* GPUs, then in one iteration
each GPU will perform forward and backward on *b/k* examples. The
gradients are then summed over all GPUs before updating the model.

### How to Use

> To use GPUs, we need to compile MXNet with GPU support. For
> example, set `USE_CUDA=1` in `config.mk` before `make`. (see
> [MXNet installation guide](http://mxnet.io/install/index.html) for more options).

If a machine has one or more GPU cards installed,
then each card is labeled by a number starting from 0.
To use a particular GPU, one can either
specify the context `context` in code
or pass `--gpus` at the command line.
For example, to use GPU 0 and 2 in python,
one can typically create a module with
```python
import mxnet as mx
module = mx.module.Module(context=[mx.gpu(0), mx.gpu(2)], ...)
```
while if the program accepts a `--gpus` flag (as seen in
[example/image-classification](https://github.com/dmlc/mxnet/tree/master/example/image-classification)),
then we can try
```bash
python train_mnist.py --gpus 0,2 ...
```

### Advanced Usage
If the available GPUs are not all equally powerful,
we can partition the workload accordingly.
For example, if GPU 0 is 3 times faster than GPU 2,
then we might use the workload option `work_load_list=[3, 1]`,
see [Module](http://mxnet.io/api/python/module/module.html#mxnet.module.Module)
for more details.

Training with multiple GPUs should yield the same results
as training on a single GPU if all other hyper-parameters are the same.f
In practice, the results may exhibit small differences,
owing to the randomness of I/O (random order or other augmentations),
weight initialization with different seeds, and CUDNN.

We can control on which devices the gradient is aggregated
and on which device the model is updated via [`KVStore`](http://mxnet.io/api/python/kvstore/kvstore.html),
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

## Distributed training with multiple devices across machines
Refer [Distributed training](https://mxnet.incubator.apache.org/versions/master/faq/distributed_training.html)
for information on how distributed training works and how to use it.
