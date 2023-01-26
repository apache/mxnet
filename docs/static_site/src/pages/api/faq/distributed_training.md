---
layout: page_category
title:  Distributed Training in MXNet
category: faq
faq_c: Deployment Environments
question: How to do distributed training using MXNet on AWS?
permalink: /api/faq/distributed_training
---
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

# Distributed Training in MXNet
MXNet supports distributed training enabling us to leverage multiple machines for faster training.
In this document, we describe how it works, how to launch a distributed training job and
some environment variables which provide more control.

## Types of Parallelism
There are two ways in which we can distribute the workload of training a neural network across multiple devices (can be either GPU or CPU).
The first way is *data parallelism*, which refers to the case where each device stores a complete copy of the model.
Each device works with a different part of the dataset, and the devices collectively update a shared model.
These devices can be located on a single machine or across multiple machines.
In this document, we describe how to train a model with devices distributed across machines in a data parallel way.

When models are so large that they don't fit into device memory, then a second way called *model parallelism* is useful.
Here, different devices are assigned the task of learning different parts of the model.
Currently, MXNet supports Model parallelism in a single machine only. Refer [Training with multiple GPUs using model parallelism](model_parallel_lstm) for more on this.

## How Does Distributed Training Work?
The following concepts are key to understanding distributed training in MXNet:
### Types of Processes
MXNet has three types of processes which communicate with each other to accomplish training of a model.
- Worker: A worker node actually performs training on a batch of training samples.
Before processing each batch, the workers pull weights from servers.
The workers also send gradients to the servers after each batch.
Depending on the workload for training a model, it might not be a good idea to run multiple worker processes on the same machine.
- Server: There can be multiple servers which store the model's parameters, and communicate with workers.
A server may or may not be co-located with the worker processes.
- Scheduler: There is only one scheduler. The role of the scheduler is to set up the cluster. This includes waiting for messages that each node has come up and which port the node is listening on.
The scheduler then lets all processes know about every other node in the cluster, so that they can communicate with each other.

### KV Store
MXNet provides a key-value store, which is a critical component used for multi-device training. The communication of parameters across devices on a single machine, as well as across multiple machines, is relayed through one or more servers with a key-value store for the parameters. Each value in this store is represented by a key and value, where each parameter array in the network is assigned a key, and value refers to the weights of that parameter array. Workers `push` gradients after processing a batch, and `pull` updated weights before processing a new batch.
We can also pass in optimizers for the KVStore to use while updating each weight. Optimizers like Stochastic Gradient Descent define an update rule,
essentially a mathematical formula to compute the new weight based on the old weight, gradient, and some parameters.

If you are using a Gluon Trainer object or the Module API,
it uses a kvstore object internally to aggregate gradients from multiple devices on the same machine as well as across different machines.

Although the API remains the same whether or not multiple machines are being used,
the notion of kvstore server exists only during distributed training.
In this case, each `push` and `pull` involves communication with the kvstore servers. When there are multiple devices on a single machine, gradients from these devices are first aggregated on the machine and then sent to the servers.
Note that we need to compile MXNet with the build flag `USE_DIST_KVSTORE=1` to use distributed training.

The distributed mode of KVStore is enabled by calling `mxnet.kvstore.create` function
with a string argument which contains the word `dist` as follows:
> kv = mxnet.kvstore.create('dist_sync')

Refer [KVStore API]({{'/api/python/docs/api/kvstore/index.html#mxnet.kvstore.KVStore'|relative_url}}) for more information about KVStore.

### Distribution of Keys
Each server doesn't necessarily store all the keys or parameter arrays.
Parameters are distributed across different servers. The decision of which server stores a particular key is made at random.
This distribution of keys across different servers is handled transparently by the KVStore.
It ensures that when a key is pulled, that request is sent to the server which has the corresponding value.
If the value of some key is very large, it may be sharded across different servers. This means that different servers hold different parts of the value.
Again, this is handled transparently so that the worker does not have to do anything different.
The threshold for this sharding can be controlled with the environment variable `MXNET_KVSTORE_BIGARRAY_BOUND`.
See [environment variables](#environment-variables) for more details.

### Split training data
When running distributed training in data parallel mode, we want each machine to be working on different parts of the dataset.

For data parallel training on a single worker,
we can use `mxnet.gluon.utils.split_and_load` to split a batch of samples provided by the data iterator, and then load each part of the batch on the device which will process it.

In the case of distributed training though, we would need to divide the dataset into `n` parts at the beginning, so that each worker gets a different part. Each worker can then use `split_and_load` to again divide that part of the dataset across different devices on a single machine.

Typically, this split of data for each worker happens through the data iterator,
on passing the number of parts and the index of parts to iterate over.
Some iterators in MXNet that support this feature are [mxnet.io.MNISTIterator](/api/python/docs/api/mxnet/io/index.html?MNISTIter#mxnet.io.MNISTIter) and [mxnet.io.ImageRecordIter](api/python/docs/api/mxnet/io/index.html?imagerecorditer#mxnet.io.ImageRecordIter).
If you are using a different iterator, you can look at how the above iterators implement this.
We can use the kvstore object to get the number of workers (`kv.num_workers`) and rank of the current worker (`kv.rank`).
These can be passed as arguments to the iterator.
You can look at [example/gluon/image_classification.py](https://github.com/apache/mxnet/blob/master/example/gluon/image_classification.py)
to see an example usage.

### Updating weights
KVStore server supports two modes, one which aggregates the gradients and updates the weights using those gradients, and second where the server only aggregates gradients. In the latter case, when a worker process pulls from kvstore, it gets the aggregated gradients. The worker then uses these gradients and applies the weights locally.

When using Gluon there is an option to choose between these modes by passing `update_on_kvstore` variable when you create the [Trainer](/api/python/docs/api/gluon/trainer.html) object like this:

```
trainer = gluon.Trainer(net.collect_params(), optimizer='sgd',
                        optimizer_params={'learning_rate': opt.lr,
                                          'wd': opt.wd,
                                          'momentum': opt.momentum,
                                          'multi_precision': True},
                        kvstore=kv,
                        update_on_kvstore=True)
```

When using the symbolic interface, it performs the weight updates on the server without the user having to do anything special.

### Different Modes of Distributed Training
Distributed training itself is enabled when kvstore creation string contains the word `dist`.

Different modes of distributed training can be enabled by using different types of kvstore.

- `dist_sync`: In synchronous distributed training, all workers use the same synchronized set of model parameters at the start of every batch.
This means that after each batch, the server waits to receive gradients from each worker before it updates the model parameters.
This synchronization comes at a cost because the worker pulling parameters would have to wait till the server finishes this process.
In this mode, if a worker crashes, then it halts the progress of all workers.

- `dist_async`: In asynchronous distributed training, the server receives gradients from one worker and immediately updates its store, which it uses to respond to any future pulls.
This means that a worker who finishes processing a batch can pull the current parameters from server and start the next batch,
even if other workers haven't finished processing the earlier batch.
This is faster than `dist_sync` because there is no cost of synchronization, but can take more epochs to converge.
The update of weights is atomic, meaning no two updates happen on the same weight at the same time. However, the order  of updates is not guaranteed.
In `async` mode, it is required to pass an optimizer because in the absence of an optimizer kvstore would replace the stored weights with received weights and this doesn't make sense for training in asynchronous mode. Hence, when using Gluon with `async` mode we need to set `update_on_kvstore` to `True`.

- `dist_sync_device`: Same as `dist_sync` except that when there are multiple GPUs being used on each node,
this mode aggregates gradients and updates weights on GPU while dist_sync does so on CPU memory.
This is faster than `dist_sync` because it reduces expensive communication between GPU and CPU, but it increases memory usage on GPU.

- `dist_async_device` : The analogue of `dist_sync_device` but in asynchronous mode.


### Gradient Compression
When communication is expensive, and the ratio of computation time to communication time is low, communication can become a bottleneck.
In such cases, gradient compression can be used to reduce the cost of communication, thereby speeding up training.
Refer [Gradient compression]({{'/api/faq/gradient_compression'|relative_url}}) for more details.

Note: For small models when the cost of computation is much lower than cost of communication,
distributed training might actually be slower than training on a single machine because of the overhead of communication and synchronization.

## How to Start Distributed Training?
MXNet provides a script tools/launch.py to make it easy to launch a distributed training job. This supports various types of cluster resource managers like `ssh`, `mpirun`, `yarn` and `sge`.
If you already have one of these clusters setup, you can skip the next section on setting up a cluster.
If you want to use a type of cluster not mentioned above, skip ahead to Manually launching jobs section.

### Setting up the Cluster
An easy way to set up a cluster of EC2 instances for distributed deep learning is by using the [AWS CloudFormation template](https://github.com/awslabs/deeplearning-cfn).
If you can not use the above, this section will help you manually set up a cluster of instances
to enable you to use `ssh` for launching a distributed training job.
Let us denote one machine as the `master` of the cluster through which we will launch and monitor the distributed training on all machines.

If the machines in your cluster are a part of a cloud computing platform like AWS EC2, then your instances should be using key-based authentication already.
Ensure that you create all instances using the same key, say `mxnet-key` and in the same security group.
Next, we need to ensure that master has access to all other machines in the cluster through `ssh` by
adding this key to [ssh-agent](https://en.wikipedia.org/wiki/Ssh-agent) and forwarding it to master when we log in. This will make `mxnet-key` the default key on master.

```
ssh-add .ssh/mxnet-key
ssh -A user@MASTER_IP_ADDRESS
```


If your machines use passwords for authentication, see [here](https://help.ubuntu.com/community/SSH/OpenSSH/Keys) for instructions on setting up password-less authentication between machines.


It is easier if all these machines have a shared file system so that they can access the training script. One way is to use [Amazon Elastic File System](https://aws.amazon.com/efs) to create your network file system.
The options in the following command are the recommended options when mounting an AWS Elastic File System.

```
sudo mkdir efs && sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 NETWORK_FILE_SYSTEM_IP:/ efs
```

Tip: You might find it helpful to store large datasets on S3 for easy access from all machines in the cluster. Refer [Using data from S3 for training]({{'/api/faq/s3_integration'|relative_url}}) for more information.

### Using Launch.py
MXNet provides a script [tools/launch.py](https://github.com/apache/mxnet/blob/master/tools/launch.py) to make it easy to launch distributed training on a cluster with `ssh`, `mpi`, `sge` or `yarn`.
You can fetch this script by cloning the mxnet repository.

```
git clone --recursive https://github.com/apache/mxnet
```

#### Example
Let us consider training a VGG11 model on the CIFAR10 dataset using [example/gluon/image_classification.py](https://github.com/apache/mxnet/blob/master/tools/launch.py).
```
cd example/gluon/
```
On a single machine, we can run this script as follows:
```
python image_classification.py --dataset cifar10 --model vgg11 --epochs 1
```

For distributed training of this example, we would do the following:

If the mxnet directory which contains the script `image_classification.py` is accessible to all machines in the cluster (for example if they are on a network file system), we can run:
```
../../tools/launch.py -n 3 -H hosts --launcher ssh python image_classification.py --dataset cifar10 --model vgg11 --epochs 1 --kvstore dist_sync
```

If the directory with the script is not accessible from the other machines in the cluster, then we can synchronize the current directory to all machines.
```
../../tools/launch.py -n 3 -H hosts --launcher ssh --sync-dst-dir /tmp/mxnet_job/ python image_classification.py --dataset cifar10 --model vgg11 --epochs 1 --kvstore dist_sync
```

> Tip: If you don't have a cluster ready and still want to try this out, pass the option `--launcher local` instead of `ssh`

#### Options
Here, launch.py is used to submit the distributed training job. It takes the following options:
- `-n` denotes the number of worker nodes to be launched.
- `-s` denotes the number of server nodes to be launched.
If it is not specified, it is taken to be equal to the number of worker nodes.
The script tries to cycle through the hosts file to launch the servers and workers.
For example, if you have 5 hosts in the hosts file and you passed `n` as 3 (and nothing for `s`).
The script will launch a total of 3 server processes,
one each for the first three hosts and launch a total of 3 worker processes, one each for the fourth, fifth and first host.
If the hosts file has exactly `n` number of worker nodes, it will launch a server process and a worker process on each of the `n` hosts.
- `--launcher` denotes the mode of communication. The options are:
    - `ssh` if machines can communicate through ssh without passwords. This is the default launcher mode.
    - `mpi` if Open MPI is available
    - `sge` for Sun Grid Engine
    - `yarn` for Apache Yarn
    - `local` for launching all processes on the same local machine. This can be used for debugging purposes.
- `-H` requires the path of the hosts file
  This file contains IPs of the machines in the cluster. These machines should be able to communicate with each other without using passwords.
  This file is only applicable and required when the launcher mode is `ssh` or `mpi`.
  An example of the contents of the hosts file would be:
  ```
  172.30.0.172
  172.31.0.173
  172.30.1.174
  ```
- `--sync-dst-dir` takes the path of a directory on all hosts to which the current working directory will be synchronized. This only supports `ssh` launcher mode.
This is necessary when the working directory is not accessible to all machines in the cluster. Setting this option synchronizes the current directory using rsync before the job is launched.
If you have not installed MXNet system-wide
then you have to copy the folder `python/mxnet` and the file `lib/libmxnet.so` into the current directory before running `launch.py`.
For example if you are in `example/gluon`, you can do this with `cp -r ../../python/mxnet ../../lib/libmxnet.so .`. This would work if your `lib` folder contains `libmxnet.so`, as would be the case when you use make. If you use CMake, this file would be in your `build` directory.

- `python image_classification.py --dataset cifar10 --model vgg11 --epochs 1 --kvstore dist_sync`
is the command for the training job on each machine. Note the use of `dist_sync` for the kvstore used in the script.

#### Terminating Jobs
If the training job crashes due to an error or if we try to terminate the launch script while training is running,
jobs on all machines might not have terminated. In such a case, we would need to terminate them manually.
If we are using `ssh` launcher, this can be done by running the following command where `hosts` is the path of the hostfile.
```
while read -u 10 host; do ssh -o "StrictHostKeyChecking no" $host "pkill -f python" ; done 10<hosts
```

### Manually Launching Jobs
If for some reason, you do not want to use the script above to start distributed training, then this section will be helpful.
MXNet uses environment variables to assign roles to different processes and to let different processes find the scheduler.
The environment variables are required to be set correctly as follows for the training to start:
- `DMLC_ROLE`: Specifies the role of the process. This can be `server`, `worker` or `scheduler`. Note that there should only be one `scheduler`.
When `DMLC_ROLE` is set to `server` or `scheduler`, these processes start when mxnet is imported.
- `DMLC_PS_ROOT_URI`: Specifies the IP of the scheduler
- `DMLC_PS_ROOT_PORT`: Specifies the port that the scheduler listens to
- `DMLC_NUM_SERVER`: Specifies how many server nodes are in the cluster
- `DMLC_NUM_WORKER`: Specifies how many worker nodes are in the cluster

Below is an example to start all jobs locally on Linux or Mac. Note that starting all jobs on the same machine is not a good idea.
This is only to make the usage clear.

```bash
export COMMAND='python example/gluon/image_classification.py --dataset cifar10 --model vgg11 --epochs 1 --kvstore dist_sync'
DMLC_ROLE=server DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 $COMMAND &
DMLC_ROLE=server DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 $COMMAND &
DMLC_ROLE=scheduler DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 $COMMAND &
DMLC_ROLE=worker DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 $COMMAND &
DMLC_ROLE=worker DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 $COMMAND
```

For an in-depth discussion of how the scheduler sets up the cluster, you can go [here](https://blog.kovalevskyi.com/mxnet-distributed-training-explained-in-depth-part-1-b90c84bda725).

## Environment Variables
### For tuning performance
 - `MXNET_KVSTORE_REDUCTION_NTHREADS`
  Value type: Integer
  Default value: 4
  The number of CPU threads used for summing up big arrays on a single machine
  This will also be used for `dist_sync` kvstore to sum up arrays from different contexts on a single machine.
  This does not affect summing up of arrays from different machines on servers.
  Summing up of arrays for `dist_sync_device` kvstore is also unaffected as that happens on GPUs.

- `MXNET_KVSTORE_BIGARRAY_BOUND`
  Value type: Integer
  Default value: 1000000
  The minimum size of a *big array*.
  When the array size is bigger than this threshold, `MXNET_KVSTORE_REDUCTION_NTHREADS` threads are used for reduction.
  This parameter is also used as a load balancer in kvstore.
  It controls when to partition a single weight to all the servers.
  If the size of a single weight matrix is less than this bound, then it is sent to a single randomly picked server; otherwise, it is partitioned to all the servers.

- `MXNET_ENABLE_GPU_P2P` GPU Peer-to-Peer communication
  Value type: 0(false) or 1(true)
  Default value: 1
  If true, MXNet tries to use GPU peer-to-peer communication, if available on your device. This is used only when kvstore has the type `device` in it.

### Communication
- `DMLC_INTERFACE` Using a particular network interface
  Value type: Name of interface
  Example: `eth0`
  MXNet often chooses the first available network interface.
  But for machines with multiple interfaces, we can specify which network interface to use for data communication using this environment variable.

- `PS_VERBOSE` Logging communication
  Value type: 1 or 2
  Default value: (empty)
    - `PS_VERBOSE=1` logs connection information like the IPs and ports of all nodes
    - `PS_VERBOSE=2` logs all data communication information


When the network is unreliable, messages being sent from one node to another might get lost.
The training process can hang when a critical message is not successfully delivered.
In such cases, an additional ACK can be sent for each message to track its delivery.
This can be done by setting `PS_RESEND` and `PS_RESEND_TIMEOUT`
- `PS_RESEND` Retransmission for unreliable network
Value type: 0(false) or 1(true)
Default value: 0
Whether or not to enable retransmission of messages
- `PS_RESEND_TIMEOUT` Timeout for ACK to be received
Value type: Integer (in milliseconds)
Default value: 1000
If ACK is not received in `PS_RESEND_TIMEOUT` milliseconds, then the message will be resent.
