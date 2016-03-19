# Distributed Training

In this tutorial we explain how to develop distributed
training programs in MXNet and how to run them on multiple machines.

## How to Write a Distributed Program on MXNet

In distributed training, multiple machines work together to finish one training
job. Writing a distributed training program in MXNet is straightforward. It provides
a distributed key-value store named `kvstore` to hide the complexity of data
synchronization. It provides the following functions:

- `push`: Push local data, such as computed gradient, to the distributed store
- `pull`: Pull data from the distributed store, such as the newest weight
- `set_updater`: set an updater for the distributed store, which specify how the
store to merge the received data, e.g. how to update the weight using the
received gradient.

Things are even simpler if our program have the following structure:

```python
data  = mx.io.ImageRecordIter(...)
net   = mx.symbol.SoftmaxOutput(...)
model = mx.model.FeedForward.create(symbol = net, X = data, ...)
```

It reads data from an image record iterator, and train a model using a symbolic
network.  To extend it into a distributed training program, we first create a
`kvstore` and then pass it into the function `create`. The following program
modifies the above one from stochastic gradient descent (SGD) into distributed
asynchronous SGD:

```python
kv    = mx.kvstore.create('dist_sync')
model = mx.model.FeedForward.create(symbol = net, X = data, kvstore = kv, ...)
```

To have a quick test, we can simulate a distributed environment on the local
machine. The following command runs `train.py` using 2 worker nodes (and 2 server
nodes) in local. More details will be given later.

```bash
mxnet/tracker/dmlc_local.py -n 2 -s 2 python train.py
```

### Data Parallelization

On the above example, each worker processes the whole training data. It is
often not desirable for the convergence since they may compute the gradients on
the same minibatch in an iteration and therefore there is no speedup from 1
worker to multiple workers.

This problem can be solved by data parallelization, namely each worker only
reads a part of the data. The `kvstore` provides two functions to query the
worker information:

- `kvstore.num_workers` returns the number of workers.
- `kvstore.rank` returns the unique rank of the current worker, which is an
   integer in [0, `kvstore.num_workers`-1].

Furthermore, the data iterators provided in `mxnet` support to (virtually)
partition a data into multiple parts, and only reads one part from
it. Therefore, we can modify the above program to partition `data` into
`num_workers` parts and ask each worker to only read one part:

```python
data = mx.io.ImageRecordIter(num_parts = kv.num_workers, part_index = kv.rank, ...)
```

### Synchronous vs Asynchronous

The `kvstore` provides two ways to extend the (minibatch) SGD into a distributed
version. The first way uses the Bulk Synchronous Parallel (BSP) protocol, which
is called `dist_sync` on MXNet. It aggregates the gradients over all workers in
each iteration (or minibatch) before updating the weight. Assume each worker
uses (mini-)batch size *b*, and there *n* workers in total. Then `dist_sync`
will produce results similar to a single machine program with batch size
*b × n*.

Note that, due to the imperfect data partition, each worker may get slightly
different size of data. To make sure each worker computes the same number of
batches in each epoch (data pass), we need to explicitly set `epoch_size` in the
function `create` for `dist_sync` (no needs for `dist_async`). One choice is

```
epoch_size = num_examples_in_data / batch_size / kv.num_workers
```

The second way uses asynchronous updating, named `dist_async`. In this protocol,
each worker updates the weight independently with each worker. Still assume
there are *n* workers and each worker uses batch size *b*. Then `dist_async` can
be view as a single-machine SGD using batch size *b*, but in each iteration, it
may use the weight several (on average *n*) iteration ago to compute the
gradient.

Which one is better often depends on several factors. In general speaking,
`dist_async` is faster than `dist_sync` since there is no synchronization
between workers. But `dist_sync` guarantees the convergence, namely it is equal
to a single machine version with proper batch size. The convergence speed of
`dist_async`, on the other hand, is still an interesting research topic.


If using `dist_sync`, a server first aggregates the
gradients from all workers and then performances updating. While if using
`dist_async`, the server updates the weight immediately after gradients from
any one worker are received.

<!-- ### Implementation with the parameter server -->

<!-- The `kvstore` is implemented by using the [parameter server](https://github.com/dmlc/ps-lite), which -->
<!-- is a distributed framework optimized for machine learning jobs. In this -->
<!-- framework, there are multiple worker nodes and server nodes. The architecture is shown below: -->

<!-- <img src=https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/multi-node/ps_arch.png width=400/> -->

<!-- - In each iteration, a worker first reads a data batch, next **pull** the weights from the -->
<!--   servers, and then compute the gradients and **push** them to the servers. Workers -->
<!--   use several technologies, such as pre-fetching, multi-threads, and data filters, to -->
<!--   reduce the I/O overhead. -->

<!-- - A server maintains a part of the model, and **updates** the model using the -->
<!--   received gradients. It supports multiple data consistency models, with will be -->
<!--   explained later. -->

## Launch Jobs on a Cluster

MXNet provides several ways to launch jobs on a cluster with multiple machines,
including by resource managers such as `Yarn` or simply via `ssh`. [This tutorial](aws.md)
gives a step-by-step example on how to setup and run jobs on a GPU cluster at
Amazon AWS.

## Machines with multiple NIC (Network Interface Card)

For the user who's cluster is composed of machines equipped with multiple NICs, MXNet provides a way to let you choose the NIC you want to use. 
For example, your machine is equipped with two NICs, one card's name is "eth0 (ethernet device)", another one is named "ib0 (infiniband device)".
You can choose NIC which you will by following the guide below.
```
export DMLC_INTERFACE="ib0"
```
or
```
export DMLC_INTERFACE="eth0"
```
<!-- ## More Readings -->

<!-- - [Distributed training examples with results](https://github.com/dmlc/mxnet/tree/master/example/distributed-training) -->
<!-- - Research papers for the parameter server: from the algorithm aspect -->
<!--   [NIPS'14](http://www.cs.cmu.edu/~muli/file/parameter_server_nips14.pdf), from -->
<!--   the system aspect [OSDI'14](http://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf) -->
