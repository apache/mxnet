# Distributed training

MXNet supports distributed training enabling you to take advantage of multiple nodes to train models faster.
In this tutorial we describe how it works, instructions on how to launch a distributed training job and
some environment variables which provide finer control.

## Type of parallelism

There are two ways in which we can distribute the workload of training a neural network across multiple devices (which can be GPU or CPU).
The first way *data parallelism* refers to the case where each device stores a complete copy of the model.
The samples in a batch are divided across the available devices, and the devices collectively update a shared model.
These devices can be located in a single machine or across multiple machines.
In this document we describe how to train a model with devices distributed across machines in a data parallel way.

When models are so large that they don't fit into device memory, then a second way called *model parallelism* is useful.
Here, different devices are assigned the task of learning different parts of the model.
Currently MXNet supports Model parallelism in a single machine only. Refer to docs/faq/model_parallel_lstm.md for more on this.

## How it works?

The architecture of distributed training in MXNet is as follows.
#### Types of processes
MXNet has three types of processes communication between each other to accomplish training of a model together.
- Worker: A worker node actually performs training on a batch of training samples.
Before processing each batch, the workers pull weights from servers.
The workers also send gradients to the servers after each batch.
- Server: There can be multiple servers which store the model's parameters, and communicate with workers.
- Scheduler: There is only one scheduler.
The role of the scheduler is to set up the cluster.
This includes waiting for messages that each node has come up and which port the node is listening on.
The scheduler then lets all processes know about every other node in the cluster, so that they can communicate with each other.

#### Batch sizes
Note that during distributed training, batch size refers to the batch size on one machine.
If there are `n` machines, and batch size is `b`, then distributed training behaves like single node training with batch size `n*b`.
Also note that if there are g gpus on a single machine, then the batch size on each gpu would be `b/g`

#### Flow of distributed training
Say

#### Different modes of distributed training
KVStore objects handle the communication underneath to provide a simple API to push and pull the parameters of the model.
This interface is in the form of a key value store, where each parameter array is the value for a particular key.
The different modes of distributed training can be enabled by using different types of kvstores.
Distributed training itself is enabled when kvstore contains the word `dist`.

- `dist_sync` : In synchronous distributed training, all workers use the same set of model parameters to start a particular batch.
This means that after each batch, the server waits to receive gradients from each worker before it updates the model parameters.
This synchronization comes at a cost because the worker pulling parameters would have to wait till the server finishes this process.
In this mode if a worker crashes, then it halts the progress of all workers.

- `dist_async` : In asynchronous distributed training, the server receives gradients from one worker and immediately updates its store, which it uses to respond to any future pulls.
This means that a worker who finishes processing a batch can pull the current parameters from server and start the next batch,
even if other workers haven't finished processing the earlier batch.
This is faster than `dist_sync`, but can take more epochs to converge.
In `async` mode, it is required to pass an optimizer because in the absence of an optimizer kvstore would replace the stored weights with received weights and this doesn't make sense for training in asynchronous mode.
The update of weights is atomic, meaning no two updates happen on the same weight at the same time. However the order is not guaranteed.

- `dist_sync_device` : Same as dist_sync except that when there are multiple GPUs being used on each node,
this mode aggregates gradients and updates weights on GPU while dist_sync does so on CPU memory.
This is faster than `dist_sync' because it reduces expensive communication between GPU and CPU, but it increases memory usage on GPU.

- `dist_async_device` : The analogue of `dist_sync_device` but in asynchronous mode.

#### Distribution of parameter arrays
Each server doesn't necessarily store all the parameter arrays.
Arrays are distributed across different servers, the decision on which server stores a particular array is picked at random.
The worker processes are unaware of this distribution because kvstore ensures that when a particular key is being pulled, this request is sent to the server which has the corresponding value.
If the value of some key is very large, it it may be sharded across different servers.
Again, this is handled internally, so that the worker does not have to do anything different.
The threshold for this sharding can be controlled by the environment variable `MXNET_KVSTORE_BIGARRAY_BOUND`.
See environment variables section for more details.

## Launching distributed training

> Note that we need to compile MXNet with the build flag `USE_DIST_KVSTORE=1` to use distributed training.

MXNet provides a script tools/launch.py to make it easy to launch a distributed training job. This supports various types of cluster resource managers like `ssh`, `mpirun`, `yarn` and `sge`.
If you already have one of these clusters setup, you can skip ahead to Using launch.py section.
If you want to use a different kind of cluster, skip ahead to Manually launching jobs section. We describe one way to setup a cluster of machines below which communicate with each other through ssh connections.
#### Setting up the cluster

In this section we describe instructions on how to set up a cluster of instances which can communicate with each other using `ssh`. You can also





#### Using launch.py
MXNet provides a script tools/launch.py to make it easy to launch distributed training on a cluster with `ssh`, `mpi`, `sge` or `yarn`. If you don't already have a cluster, an easy way to set up a cluster of EC2 instances for distributed deep learning is using an [AWS CloudFormation template](https://github.com/awslabs/deeplearning-cfn).
##### Example usage
Let's start with an example on how to use this. Assume we are in `mxnet/example/gluon` and want to train a multi layer perceptron to classify MNIST images with the script mnist.py.
On a single machine we can run this script as
```
python mnist.py --batch-size 100
```

If the mxnet directory which contains the script mnist.py is accessible to all machines in the cluster (for example if they are on a network file system), we can run the following
```
../../tools/launch.py -n 4 --launcher ssh -H hosts python train_mnist.py --network lenet --gpus 0,1 --kv-store dist_sync_device
```
#### Options
Here, launch.py is used to submit the distributed training job. It takes the following options
- -n denotes number of worker nodes to be launched
- -s denotes number of server nodes to be launched. If it is not specified, it is taken to be equal to the number of worker nodes.
- --launcher denotes the mode of communication. The options are
    - `ssh` if machines can communicate through ssh
    - `mpi` if Open MPI is available
    - `sge` for Sun Grid Engine
    - `yarn` for Apache Yarm
    - `local` for launching all processes on the same local machine. This can be used for debugging purposes.
- -H requires the path of the hosts file<br/>
  This file contains IPs of the machines in the cluster. These machines should be able to communicate with each other. For `ssh`, passwordless authentication should be enabled. This file is only applicable and required when the launcher mode is `ssh` or `mpi`.
  An example of the contents of the hosts file would be
  ```
  172.30.0.172
  172.31.0.173
  172.30.1.174
  172.32.0.174
  ```
- --sync-dst-dir takes the path of directory on all hosts to which the current working directory will be synchronized. This only supports `ssh` launcher mode. This is necessary when the working directory is not accessible to all machines in the cluster. If you have not installed mxnet on the system, or are running a custom build, then you might have to copy the folder mxnet/python/mxnet and the file mxnet/lib/libmxnet.so into the current directory before starting the launch script. <br/>
For example if you are in mxnet/example/image-classification, you can do this with `cp -r ../../python/mxnet ../../lib/libmxnet.so .` Then pass `--sync_dst_dir /tmp/mxnet_job/` to synchronize current directory to given folder and then start the job.
- `python train_mnist.py --network lenet --gpus 0,1 --kv-store dist_sync_device` is the command for the training job on each machine. Note the use of `dist_sync_device` for the kvstore used in the training script.

#### Terminating jobs
If the training job crashes due to an error or we try to terminate the launch script while training is running, jobs on all machines might not have terminated and this would be need to be done manually. If we are using `ssh` launcher, this can be done by running the following command where `hosts` is the path of the hostfile.
```
while read -u 10 host; do ssh -o "StrictHostKeyChecking no" $host "pkill -f python" ; done 10<hosts
```

### Manually launching jobs
If for some reason, you do not want to use the script above to start distributed training, then you can do so by keeping in mind the following. MXNet uses environment variables to assign roles to different processes, and to let different processes find the scheduler. The environment variables are required to be set correctly for the training to start:
- `DMLC_ROLE` : Specifies the role of the process. This can be `server`, `worker` or `scheduler`. Note that there should only be one `scheduler`
- `DMLC_PS_ROOT_URI` : Specifies the IP of the scheduler
- `DMLC_PS_ROOT_PORT` : Specifies the port that the scheduler listens to
- `DMLC_NUM_SERVER` : Specifies how many server nodes are in the cluster
- `DMLC_NUM_WORKER` : Specifies how many worker nodes are in the cluster

Below is an example to start all jobs locally on Linux or Mac. Note that starting all jobs on the same machine is not a good idea. This is only to make the usage clear.
```
export COMMAND=python example/gluon/mnist.py --kv-store dist_async
DMLC_ROLE=server DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 COMMAND &
DMLC_ROLE=server DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 COMMAND &
DMLC_ROLE=scheduler DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 COMMAND &
DMLC_ROLE=worker DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 COMMAND &
DMLC_ROLE=worker DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 COMMAND
```
For an in-depth discussion of how the cluster is set up, you can go [here](https://blog.kovalevskyi.com/mxnet-distributed-training-explained-in-depth-part-1-b90c84bda725)
## Environment variables
#### For tuning performance
 - `MXNET_KVSTORE_REDUCTION_NTHREADS`
  Values: Int
  Default=4
  The number of CPU threads used for summing up big arrays.

- `MXNET_KVSTORE_BIGARRAY_BOUND`
  Values: Int
  Default=1000000
  The minimum size of a “big array”.
  When the array size is bigger than this threshold, `MXNET_KVSTORE_REDUCTION_NTHREADS` threads are used for reduction.
  This parameter is also used as a load balancer in kvstore. It controls when to partition a single weight to all the servers. If the size of a single weight is less than this bound then, it is sent to a single randomly picked server otherwise it is partitioned to all the servers.

- `MXNET_ENABLE_GPU_P2P` GPU Peer-to-Peer communication
  Values: 0(false) or 1(true)
  Default: 1
  If true, MXNet tries to use GPU peer-to-peer communication, if available on your device. This is used only when kvstore has the type `device` in it.

#### Communication
- `DMLC_INTERFACE` Using a particular network interface
  Values: Interface name.
  Example `eth0`
  MXNet often chooses the first available network interface. But for machines with multiple interfaces, we can specify which network interface to use for data communication using this environment variable.

- `PS_VERBOSE` Logging communication
    - `PS_VERBOSE=1` logs connection information like the IPs and ports of all nodes
    - `PS_VERBOSE=2` logs all data communication information

When the network is unreliable, messages being sent from one node to another might disappear. The training process can hang when a critical message is not successfully devlivered. In such cases, an additional ACK can be sent for each message to track its delivery. This can be done by setting `PS_RESEND` and `PS_RESEND_TIMEOUT`
- `PS_RESEND` Retransmission for unreliable network
Values: 0(false) or 1(true)
Default: 0
Whether or not to enable retransmission of messages
- `PS_RESEND_TIMEOUT` Timeout for ACK to be received
Values: Int (in milliseconds)
Default: 1000
If ACK is not received for `PS_RESEND_TIMEOUT` milliseconds, then message will be resent.