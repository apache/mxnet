<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# MXNet + Horovod
Horovod is a distributed training framework for MXNet, TensorFlow, Keras, and PyTorch. The goal of Horovod is to make distributed Deep Learning fast and easy to use. Horovod is created at Uber and currently hosted by the [Linux Foundation Deep Learning](https://lfdl.io) (LF DL). For details about who's involved and how Horovod plays a role, read the LF DL [announcement](https://lfdl.io/press/2018/12/13/lf-deep-learning-welcomes-horovod-distributed-training-framework-as-newest-project/).

Since 0.16.0 release, MXNet is supported in Horovod. 

## Difference between Horovod and Parameter Server
Horovod uses ring allreduce algorithm to communicate parameters between workers. There is no server to store 
and aggregate parameters. The data communication between workers does not depend on the number of workers and therefore scales well when the number of workers is large the network bandwidth becomes bottleneck.

# Install

**Note**: we recommend users to build MXNet from source following this [guide](https://mxnet.incubator.apache.org/install/build_from_source.html) when running Horovod with MXNet on a Linux OS with GCC version 5.X and above. The MXNet shared library distributed through MXNet pip package is currently built using GCC 4.8.4. If we build and install Horovod on a Linux OS with GCC 5.X+ with MXNet pip package, we will hit segmentation fault due to std::function definition change from GCC [4.X](https://github.com/gcc-mirror/gcc/blob/gcc-4_8_4-release/libstdc++-v3/include/std/functional#L2069) to GCC [5.X](https://github.com/gcc-mirror/gcc/blob/gcc-5_4_0-release/libstdc++-v3/include/std/functional#L1854).

To install Horovod:

1. Install [Open MPI](https://www.open-mpi.org/) or another MPI implementation.

Steps to install Open MPI are listed [here](https://www.open-mpi.org/faq/?category=building#easy-build).

**Note**: Open MPI 3.1.3 has an issue that may cause hangs.  It is recommended
to downgrade to Open MPI 3.1.2 or upgrade to Open MPI 4.0.0.

2. Install the `horovod` pip package.

```bash
$ pip install horovod
```

This basic installation is good for laptops and for getting to know Horovod.
If you're installing Horovod on a server with GPUs, read the [Horovod on GPU](docs/gpus.md) page.
If you want to use Docker, read the [Horovod in Docker](docs/docker.md) page.

# Usage

To use MXNet with Horovod, make the following additions to your training script:

1. Run `hvd.init()`.

2. Pin a server GPU to the context using `context = mx.gpu(hvd.local_rank())`.
    With the typical setup of one GPU per process, this can be set to *local rank*. In that case, the first process on 
    the server will be allocated the first GPU, second process will be allocated the second GPU and so forth.

3. Scale the learning rate by number of workers. Effective batch size in synchronous distributed training is scaled by
    the number of workers. An increase in learning rate compensates for the increased batch size.

4. Wrap optimizer in `hvd.DistributedOptimizer`.  The distributed optimizer delegates gradient computation
    to the original optimizer, averages gradients using *allreduce* or *allgather*, and then applies those averaged
    gradients.

5. Add `hvd.broadcast_parameters` to broadcast initial variable states from rank 0 to all other processes.
    This is necessary to ensure consistent initialization of all workers when training is started with random weights or
    restored from a checkpoint. 

# Example

Here we provide the skeleton of a typical script to train a model using MXNet with Horovod.
The examples are written in Gluon and Module APIs respectively.
See the full examples in [MINST](mxnet_mnist.py) and [ImageNet](mxnet_imagenet_resnet50.py).

## Gluon API
```python
from mxnet import autograd, gluon
import mxnet as mx
import horovod.mxnet as hvd

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank
context = mx.gpu(hvd.local_rank())
num_workers = hvd.size()

# Build model
model = ...
model.hybridize()

# Define hyper parameters
optimizer_params = ...

# Add Horovod Distributed Optimizer
opt = mx.optimizer.create('sgd', **optimizer_params)
opt = hvd.DistributedOptimizer(opt)

# Initialize parameters
model.initialize(initializer, ctx=context)

# Fetch and broadcast parameters
params = model.collect_params()
if params is not None:
    hvd.broadcast_parameters(params, root_rank=0)

# Create trainer and loss function
trainer = gluon.Trainer(params, opt, kvstore=None)
loss_fn = ...

# Train model
for epoch in range(num_epoch):
    train_data.reset()
    for nbatch, batch in enumerate(train_data, start=1):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=[context],
                                          batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=[context],
                                           batch_axis=0)
        with autograd.record():
            outputs = [model(x.astype(dtype, copy=False)) for x in data]
            loss = [loss_fn(yhat, y) for yhat, y in zip(outputs, label)]
        for l in loss:
            l.backward()
        trainer.step(batch_size)
```

## Module API
```python
import mxnet as mx
import horovod.mxnet as hvd

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank
context = mx.gpu(hvd.local_rank())
num_workers = hvd.size()

# Build model
model = ...

# Define hyper parameters
optimizer_params = ...

# Add Horovod Distributed Optimizer
opt = mx.optimizer.create('sgd', **optimizer_params)
opt = hvd.DistributedOptimizer(opt)

# Initialize parameters
initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in",
                             magnitude=2)
model.bind(data_shapes=train_data.provide_data,
           label_shapes=train_data.provide_label)
model.init_params(initializer)

# Fetch and broadcast parameters
(arg_params, aux_params) = model.get_params()
if arg_params:
    hvd.broadcast_parameters(arg_params, root_rank=0)
if aux_params:
    hvd.broadcast_parameters(aux_params, root_rank=0)
model.set_params(arg_params=arg_params, aux_params=aux_params)

# Train model
model.fit(train_data,
          kvstore=None,
          optimizer=opt,
          num_epoch=num_epoch)
```

# Running Horovod

The example commands below show how to run distributed training. See the [Running Horovod](docs/running.md)
page for more instructions, including RoCE/InfiniBand tweaks and tips for dealing with hangs.

1. To run on a machine with 4 GPUs:

```bash
$ mpirun -np 4 \
    -H localhost:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py
```

2. To run on 4 machines with 4 GPUs each:

```bash
$ mpirun -np 16 \
    -H server1:4,server2:4,server3:4,server4:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py
```