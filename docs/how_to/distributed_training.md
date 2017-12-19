# Distributed training
MXNet supports distributed training enabling us to leverage multiple machines for faster training.
In this document, we describe how it works, how to launch a distributed training job and
some environment variables which provide more control.

## Type of parallelism

There are two ways in which we can distribute the workload of training a neural network across multiple devices (can be either GPU or CPU).
The first way *data parallelism* refers to the case where each device stores a complete copy of the model.
Each device works with a different part of the dataset, and the devices collectively update a shared model.
These devices can be located in a single machine or across multiple machines.
In this document, we describe how to train a model with devices distributed across machines in a data parallel way.

When models are so large that they don't fit into device memory, then a second way called *model parallelism* is useful.
Here, different devices are assigned the task of learning different parts of the model.
Currently, MXNet supports Model parallelism in a single machine only. Refer to docs/faq/model_parallel_lstm.md for more on this.

## How it works?

The architecture of distributed training in MXNet is as follows.
#### Types of processes
MXNet has three types of processes which communicate with each other to together accomplish training of a model.
- Worker: A worker node actually performs training on a batch of training samples.
Before processing each batch, the workers pull weights from servers.
The workers also send gradients to the servers after each batch.
Depending on the workload for training a model, it might not be a good idea to run multiple worker processes on the same machine.
- Server: There can be multiple servers which store the model's parameters, and communicate with workers.
A server may or may not be co-located with the worker processes.
- Scheduler: There is only one scheduler.
The role of the scheduler is to set up the cluster.
This includes waiting for messages that each node has come up and which port the node is listening on.
The scheduler then lets all processes know about every other node in the cluster, so that they can communicate with each other.

#### KV Store
MXNet provides a key-value store, which is a critical component used for multi-device and distributed training.
It provides a push and pull API for workers to communicate the parameters of the models. It stores a parameter value for each key.
Workers `push` gradients after processing a batch, and `pull` updated weights before processing a new batch.
We can also pass in optimizers for the KVStore to use while updating each weight. Optimizers like Stochastic Gradient Descent define update rules,
essentially a mathematical formula to compute the new weight based on the old weight, gradient, and some parameters.

If you are using a Gluon Trainer object or the Module API,
it uses a kvstore object internally to aggregate gradients from multiple devices on the same machine as well as different machines.

Although the API remains the same whether or not multiple machines are being used,
the notion of kvstore server exists only during distributed training.
In this case, each `push` and `pull` involves communication with the kvstore servers.
Note that we need to compile MXNet with the build flag `USE_DIST_KVSTORE=1` to use distributed training.

KVStore can be started in distributed mode, by passing a create string which contains the word `dist`
> kv = mxnet.kvstore.create('dist')

Apart from push and pull, kvstore also allows us to fetch the number of workers and the rank of the current worker. Refer [this page](https://mxnet.incubator.apache.org/api/python/kvstore.html) for KVStore API.

#### Data iterators
When running distributed training,
we want the data iterators on each machine to be working on different parts of the dataset.
Let's look at the example in `example/gluon/image_classification.py` to understand how this is done.
For data parallel training on a single worker,
we can use `mxnet.gluon.utils.split_and_load` to split a batch of samples provided by the data iterator, and then load each part of the batch on the device which will process it.
In the case of distributed training, one way to ensure that different workers
process different samples is to divide the dataset into `n` parts at the beginning, one for each worker.
Within the part of the dataset each worker has, we can continue to split as before for each device on that worker.


Typically, this split of data for each worker happens through the data iterator,
on passing the number of parts and the index of parts to iterate over.
Some iterators in MXNet that support this feature are mxnet.io.MNISTIterator and mxnet.io.ImageRecordIter.
If you are using a different iterator, you can look at how the above iterators implement this.

#### Different modes of distributed training
Different modes of distributed training can be enabled by using different types of kvstore.
Distributed training itself is enabled when kvstore contains the word `dist`.

- `dist_sync`: In synchronous distributed training, all workers use the same synchronized set of model parameters at the start of every batch.
This means that after each batch, the server waits to receive gradients from each worker before it updates the model parameters.
This synchronization comes at a cost because the worker pulling parameters would have to wait till the server finishes this process.
In this mode, if a worker crashes, then it halts the progress of all workers.

- `dist_async`: In asynchronous distributed training, the server receives gradients from one worker and immediately updates its store, which it uses to respond to any future pulls.
This means that a worker who finishes processing a batch can pull the current parameters from server and start the next batch,
even if other workers haven't finished processing the earlier batch.
This is faster than `dist_sync` but can take more epochs to converge.
In `async` mode, it is required to pass an optimizer because in the absence of an optimizer kvstore would replace the stored weights with received weights and this doesn't make sense for training in asynchronous mode.
The update of weights is atomic, meaning no two updates happen on the same weight at the same time. However, the order  of updates is not guaranteed.

- `dist_sync_device`: Same as dist_sync except that when there are multiple GPUs being used on each node,
this mode aggregates gradients and updates weights on GPU while dist_sync does so on CPU memory.
This is faster than `dist_sync' because it reduces expensive communication between GPU and CPU, but it increases memory usage on GPU.

- `dist_async_device` : The analogue of `dist_sync_device` but in asynchronous mode.

#### Distribution of parameter arrays
Each server doesn't necessarily store all the parameter arrays.
Arrays are distributed across different servers, the decision on which server stores a particular array is picked at random.
The worker processes are unaware of this distribution because kvstore ensures that when a particular key is being pulled, this request is sent to the server which has the corresponding value.
If the value of some key is very large, it may be sharded across different servers.
Again, this is handled internally, so that the worker does not have to do anything different.
The threshold for this sharding can be controlled with the environment variable `MXNET_KVSTORE_BIGARRAY_BOUND`.
See environment variables section below for more details.

## Launching distributed training
MXNet provides a script tools/launch.py to make it easy to launch a distributed training job. This supports various types of cluster resource managers like `ssh`, `mpirun`, `yarn` and `sge`.
If you already have one of these clusters setup, you can skip the next section on setting up a cluster.
If you want to use a type of cluster not mentioned above, skip ahead to Manually launching jobs section.

#### Setting up the cluster
An easy way to set up a cluster of EC2 instances for distributed deep learning is using an [AWS CloudFormation template](https://github.com/awslabs/deeplearning-cfn).
In this section, we describe how to manually set up a cluster of instances which can communicate with each other using `ssh`, if you can not use the above.
Let us denote one machine as the `master` of the cluster, through which we will launch and monitor the distributed training machine.

If the machines in your cluster are a part of a cloud computing platform like AWS EC2, then your instances should be using key-based authentication already.
In that case, each machine needs to have the key to access the other machines in the cluster as the default key `id_rsa` or `id_dsa`. This can be done as below.
Ensure you create all instances using the same key, say `mxnet-key`.
Now while logging into master, we can add this key to ssh-agent and have it forwarded to the master instance.

```
ssh-add .ssh/mxnet-key
ssh -A user@MASTER_IP_ADDRESS
```

If your machines use passwords for authentication, see [here](https://help.ubuntu.com/community/SSH/OpenSSH/Keys) on how to set up password-less authentication between machines.


It is easier if all these machines have a shared file system so that they can access the training script. One way is to use Amazon Elastic File System to create your network file system.
The options in the next command are the recommended options for loading AWS EFS.

```
sudo mkdir efs && sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 NETWORK_FILE_SYSTEM_IP:/ efs
```
With this, you have a cluster of machines which can communicate with each other
#### Using launch.py
MXNet provides a script tools/launch.py to make it easy to launch distributed training on a cluster with `ssh`, `mpi`, `sge` or `yarn`.
Fetch this script by cloning the mxnet repository.

```
git clone --recursive https://github.com/apache/incubator-mxnet
```

##### Example
Let's start with an example on how to use this. Let us consider training a VGG11 model on the CIFAR10 dataset using the image classification example for gluon.
```
cd example/gluon/
```
On a single machine, we can run this script as
```
python image_classification.py --dataset cifar10 --model vgg11 --num-epochs 1
```

For distributed training of this example, we would do the following.

If the mxnet directory which contains the script image_classification.py is accessible to all machines in the cluster (for example if they are on a network file system), we can run:
```
../../tools/launch.py -n 3 -H hosts --launcher ssh python image_classification.py --dataset cifar10 --model vgg11 --num-epochs 1 --kvstore dist_sync
```

If the directory with the script is not accessible from the other machines in the cluster then we can synchronize the current directory to all machines.
```
../../tools/launch.py -n 3 -H hosts --launcher ssh --sync-dst-dir /tmp/mxnet_job/ python image_classification.py --dataset cifar10 --model vgg11 --num-epochs 1 --kvstore dist_sync
```

> Tip: If you don't have a cluster ready and still want to try this out, pass the option `--launcher local` instead of `ssh`

#### Options
Here, launch.py is used to submit the distributed training job. It takes the following options
- `-n` denotes the number of worker nodes to be launched
- `-s` denotes the number of server nodes to be launched.
If it is not specified, it is taken to be equal to the number of worker nodes.
The script tries to cycle through the hosts file to launch the servers and workers.
For example, if you have 5 hosts in the hosts file and you passed `n` as 3 (and nothing for `s`).
The script will launch a total of 3 server processes,
one each for the first three hosts and launch a total of 3 worker processes, one each for the fourth, fifth and first host.
If the hosts file has exactly `n` number of worker nodes, it will launch a server and worker process on each of the `n` hosts.
- `--launcher` denotes the mode of communication. The options are
    - `ssh` if machines can communicate through ssh without passwords. This is the default launcher mode.
    - `mpi` if Open MPI is available
    - `sge` for Sun Grid Engine
    - `yarn` for Apache Yarn
    - `local` for launching all processes on the same local machine. This can be used for debugging purposes.
- `-H` requires the path of the hosts file
  This file contains IPs of the machines in the cluster. These machines should be able to communicate with each other without using passwords.
  This file is only applicable and required when the launcher mode is `ssh` or `mpi`.
  An example of the contents of the hosts file would be
  ```
  172.30.0.172
  172.31.0.173
  172.30.1.174
  ```
- `--sync-dst-dir` takes the path of a directory on all hosts to which the current working directory will be synchronized. This only supports `ssh` launcher mode.
This is necessary when the working directory is not accessible to all machines in the cluster. Setting this option synchronizes the current directory using rsync, before the job is launched.

If you have not installed MXNet system-wide,
then you have to copy the folder `python/mxnet` and the file `lib/libmxnet.so` into the current directory before running `launch.py`,
For example if you are in `example/gluon`, you can do this with `cp -r ../../python/mxnet ../../lib/libmxnet.so .`

Then pass
- `python image_classification.py --dataset cifar10 --model vgg11 --num-epochs 1 --kvstore dist_sync`
is the command for the training job on each machine.
Note the use of `dist_sync` for the kvstore used in the script.

#### Terminating jobs
If the training job crashes due to an error or if we try to terminate the launch script while training is running, jobs on all machines might not have terminated. In such a case, we would need to terminate them manually. If we are using `ssh` launcher, this can be done by running the following command where `hosts` is the path of the hostfile.
```
while read -u 10 host; do ssh -o "StrictHostKeyChecking no" $host "pkill -f python" ; done 10<hosts
```

### Manually launching jobs
If for some reason, you do not want to use the script above to start distributed training, then this section will be helpful. MXNet uses environment variables to assign roles to different processes and to let different processes find the scheduler. The environment variables are required to be set correctly for the training to start:
- `DMLC_ROLE`: Specifies the role of the process. This can be `server`, `worker` or `scheduler`. Note that there should only be one `scheduler`.
When `DMLC_ROLE` is set to `server` or `scheduler`, these processes start when mxnet is imported.
- `DMLC_PS_ROOT_URI`: Specifies the IP of the scheduler
- `DMLC_PS_ROOT_PORT`: Specifies the port that the scheduler listens to
- `DMLC_NUM_SERVER`: Specifies how many server nodes are in the cluster
- `DMLC_NUM_WORKER`: Specifies how many worker nodes are in the cluster

Below is an example to start all jobs locally on Linux or Mac. Note that starting all jobs on the same machine is not a good idea. This is only to make the usage clear.
```
export COMMAND=python example/gluon/mnist.py --dataset cifar10 --model vgg11 --num-epochs 1 --kv-store dist_async
DMLC_ROLE=server DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 COMMAND &
DMLC_ROLE=server DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 COMMAND &
DMLC_ROLE=scheduler DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 COMMAND &
DMLC_ROLE=worker DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 COMMAND &
DMLC_ROLE=worker DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9092 DMLC_NUM_SERVER=2 DMLC_NUM_WORKER=2 COMMAND
```
For an in-depth discussion of how the scheduler sets up the cluster, you can go [here](https://blog.kovalevskyi.com/mxnet-distributed-training-explained-in-depth-part-1-b90c84bda725).


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

When the network is unreliable, messages being sent from one node to another might disappear. The training process can hang when a critical message is not successfully delivered. In such cases, an additional ACK can be sent for each message to track its delivery. This can be done by setting `PS_RESEND` and `PS_RESEND_TIMEOUT`
- `PS_RESEND` Retransmission for unreliable network
Values: 0(false) or 1(true)
Default: 0
Whether or not to enable retransmission of messages
- `PS_RESEND_TIMEOUT` Timeout for ACK to be received
Values: Int (in milliseconds)
Default: 1000
If ACK is not received for `PS_RESEND_TIMEOUT` milliseconds, then the message will be resent.