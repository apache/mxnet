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

# Distributed Training using MXNet with Horovod 
[Horovod](https://github.com/horovod/horovod) is a distributed training framework that demonstrates 
excellent scaling efficiency for dense models running on a large number of nodes. It currently 
supports mainstream deep learning frameworks such as MXNet, TensorFlow, Keras, and PyTorch. 
It is created at Uber and currently hosted by the [Linux Foundation Deep Learning](https://lfdl.io)(LF DL). 

MXNet is supported starting from Horovod 0.16.0 [release](https://eng.uber.com/horovod-pyspark-apache-mxnet-support/).

## What's New?
Compared with the standard distributed training script in MXNet which uses parameter server to 
distribute and aggregate parameters, Horovod uses ring allreduce and/or tree-based allreduce algorithm 
to communicate parameters between workers. There is no dedicated server and the communication data size 
between workers does not depend on the number of workers. Therefore, it scales well in the case where 
there are a large number of workers and network bandwidth is the bottleneck.

# Install
## Install MXNet
```bash
$ pip install mxnet
```
**Note**: The [known issue](https://github.com/horovod/horovod/issues/884) when running Horovod with MXNet on a Linux system with GCC version 5.X and above has been resolved. Please use MXNet 1.4.1 or later releases with Horovod 0.16.2 or later releases to avoid the GCC incompatibility issue. MXNet 1.4.0 release works with Horovod 0.16.0 and 0.16.1 releases with the GCC incompatibility issue unsolved.

## Install Horovod
```bash
$ pip install horovod
```

This basic installation is good for laptops and for getting to know Horovod.
If you're installing Horovod on a server with GPUs, read the [Horovod on GPU](https://github.com/horovod/horovod/blob/master/docs/gpus.rst) page.
If you want to use Docker, read the [Horovod in Docker](https://github.com/horovod/horovod/blob/master/docs/docker.rst) page.

## Install MPI
MPI is required to run distributed training with Horovod. Install [Open MPI](https://www.open-mpi.org/) or another MPI implementation.
Steps to install Open MPI are listed [here](https://www.open-mpi.org/faq/?category=building#easy-build).

**Note**: Open MPI 3.1.3 has an issue that may cause hangs.  It is recommended
to downgrade to Open MPI 3.1.2 or upgrade to Open MPI 4.0.0.

# Usage

To run MXNet with Horovod, make the following additions to your training script:

1. Run `hvd.init()`.

2. Pin the context to a processor using `hvd.local_rank()`.
    Typically, each Horovod worker is associated with one process. The local rank is a unique ID specifically
    for all processes running Horovod job on the same node.

3. Scale the learning rate by number of workers. Effective batch size in synchronous distributed training is scaled by
    the number of workers. An increase in learning rate compensates for the increased batch size.

4. Create `hvd.DistributedTrainer` with optimizer when using Gluon API or wrap optimizer in `hvd.DistributedOptimizer` when using Module API.  The distributed trainer or optimizer delegates gradient computation
    to the original optimizer, averages gradients using *allreduce*, and then applies those averaged
    gradients.

5. Add `hvd.broadcast_parameters` to broadcast initial variable states from rank 0 to all other processes.
    This is necessary to ensure consistent initialization of all workers when training is started with random weights or
    restored from a checkpoint. 

# Example

Here we provide the building blocks to train a model using MXNet with Horovod.
The full examples are in [MNIST](gluon_mnist.py) and [ImageNet](resnet50_imagenet.py).

## Gluon API
```python
from mxnet import autograd, gluon
import mxnet as mx
import horovod.mxnet as hvd

# Initialize Horovod
hvd.init()

# Set context to current process 
context = mx.cpu(hvd.local_rank()) if args.no_cuda else mx.gpu(hvd.local_rank())

num_workers = hvd.size()

# Build model
model = ...
model.hybridize()


# Create optimizer
optimizer_params = ...
opt = mx.optimizer.create('sgd', **optimizer_params)

# Create DistributedTrainer, a subclass of gluon.Trainer
trainer = hvd.DistributedTrainer(params, opt)

# Initialize parameters
model.initialize(initializer, ctx=context)

# Fetch and broadcast parameters
params = model.collect_params()
if params is not None:
    hvd.broadcast_parameters(params, root_rank=0)

# Create loss function
loss_fn = ...

# Train model
for epoch in range(num_epoch):
    train_data.reset()
    for nbatch, batch in enumerate(train_data, start=1):
        data = batch.data[0].as_in_context(context)
        label = batch.label[0].as_in_context(context)
        with autograd.record():
            output = model(data.astype(dtype, copy=False))
            loss = loss_fn(output, label)
        loss.backward()
        trainer.step(batch_size)
```

## Module API
```python
import mxnet as mx
import horovod.mxnet as hvd

# Initialize Horovod
hvd.init()

# Set context to current process
context = mx.cpu(hvd.local_rank()) if args.no_cuda else mx.gpu(hvd.local_rank())
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

The example commands below show how to run distributed training. See the 
[Running Horovod](https://github.com/horovod/horovod/blob/master/docs/running.rst)
page for more instructions.

1. To run on a machine with 4 CPUs:

```bash
$ mpirun -np 4 \
    -H localhost:4 \
    -bind-to none -map-by slot \
    python train.py
```

2. To run on 2 machines with 4 GPUs each:

```bash
$ mpirun -np 8 \
    -H server1:4,server2:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -mca pml ob1 -mca btl ^openib \
    python train.py
```
