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

# Setup

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

## On Kubernetes

Distributed MXNet jobs with Horovod can be submitted to a Kubernetes cluster via [Kubeflow MPI Operator](https://github.com/kubeflow/mpi-operator). Please refer to [this example](https://github.com/kubeflow/mpi-operator/tree/master/examples/mxnet) for details, including the Dockerfile with all the dependencies mentioned in previous sections, distributed training Python script based on Horovod, and the YAML configuration file that can be used for submitting a job on a Kubernetes cluster.

# Usage

To run MXNet with Horovod, make the following additions to your training script:

1. Run `hvd.init()`.

2. Pin the context to a processor using `hvd.local_rank()`.
    Typically, each Horovod worker is associated with one process. The local rank is a unique ID specifically
    for all processes running Horovod job on the same node.

3. Scale the learning rate by number of workers. Effective batch size in synchronous distributed training is scaled by
    the number of workers. An increase in learning rate compensates for the increased batch size.

4. Create `hvd.DistributedTrainer` with optimizer when using Gluon API.  The distributed trainer or optimizer delegates gradient computation
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

## Tuning Horovod Performance

1. To analyse horovod performance, [horovod timeline](https://github.com/horovod/horovod/blob/master/docs/timeline.rst) is a handy tool to trace and visualize the time spent on horovod operations. 

2. A few tuning knobs affect horovod runtime performance (explained [here](https://github.com/horovod/horovod/blob/master/docs/tensor-fusion.rst)). Apart from `HOROVOD_FUSION_THRESHOLD`, sometimes we find increasing `HOROVOD_CYCLE_TIME` (up to 100 ms), changing [`NCCL_ALGO`](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html#nccl-algo), and [`NCCL_MIN_NCHANNELS`](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html#nccl-min-nchannels) improves performance.

3. If you are running horovod on AWS, you can potentially leverage [EFA](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html) if your instance supports 100 Gb/s networking. To use EFA, you can refer to the [official documentation](https://docs.aws.amazon.com/eu_us/AWSEC2/latest/UserGuide/efa-start-nccl-dlami.html) for the setup instructions, and the environment variables (`-x FI_PROVIDER`, `-x FI_EFA_TX_MIN_CREDITS`) to set. Besides, you need to make sure EFA library is included in the shared library path (`-x LD_LIBRARY_PATH`).
