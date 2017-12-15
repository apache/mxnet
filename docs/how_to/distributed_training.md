# Distributed training

MXNet supports distributed training enabling you to take advantage of multiple nodes to train models faster.

## How it works?

Distributed training requires synchronization of parameters across different machines after each batch. The architecture MXNet uses is as follows.

### Types of processes
MXNet has three types of processes communication between each other to accomplish training of a model together.
- Worker: A worker node actually performs training on a batch of training samples. Before processing each batch, the workers pull weights from servers. The workers also send gradients to the servers after each batch.
- Server: There can be multiple servers which store the model's parameters, and communicate with workers.
- Scheduler: There is only one scheduler, which sets the cluster up and allows different nodes to find each other.

### Different modes of distributed training
KVStore objects handle the communication underneath to provide a simple API to push and pull the parameters of the model. This interface is in the form of a key value store, where each parameter array is the value for a particular key. The different modes of distributed training can be enabled by using different types of kvstores. Distributed training itself is enabled when kvstore contains the word `dist`.

- `dist_sync` : In synchronous distributed training, all workers use the same set of model parameters to start a particular batch. This means that after each batch, the server waits to receive gradients from each worker before it updates the model parameters. This synchronization comes at a cost because the worker pulling parameters would have to wait till the server finishes this process.

- `dist_async` : In asynchronous distributed training, the server receives gradients from one worker and immediately updates its store, which it uses to respond to any future pulls. This means that a worker who finishes processing a batch can pull the current parameters from server and start the next batch, even if other workers haven't finished processing the earlier batch. This is faster than `dist_sync`, but can take more epochs to converge. In `async` mode, it is required to pass an optimizer because in the absence of an optimizer kvstore would replace the stored weights with received weights and this doesn't make sense for training in asynchronous mode.

- `dist_sync_device` : Same as dist_sync except that when there are multiple GPUs being used on each node, this mode aggregates gradients and updates weights on GPU while dist_sync does so on CPU memory. This is faster than `dist_sync' because it reduces expensive communication between GPU and CPU.

- `dist_async_device` : The analogue of `dist_sync_device` but in asynchronous mode.

### Batch sizes
Note that during distributed training, batch size refers to the batch size on one machine. If there are `n` machines, and batch size is `b`, then distributed training behaves like single node training with batch size `n*b`. Also note that if there are g gpus on a single machine, then the batch size on each gpu would be `b/g`

### Distribution of parameter arrays
Each server doesn't necessarily store all the parameter arrays. Arrays are distributed across different servers, the decision on which server stores a particular array is picked at random. The worker processes are unaware of this distribution because kvstore ensures that when a particular key is being pulled, this request is sent to the server which has the corresponding value.
If the value of some key is very large, it it may be sharded across different servers. Again, this is handled internally, so that the worker does not have to do anything different. The threshold for this sharding can be controlled by the environment variable `MXNET_KVSTORE_BIGARRAY_BOUND`. See environment variables section for more detail.

## How to launch?
Note that we need to compile MXNet with the build flag `USE_DIST_KVSTORE=1` to use distributed training.

### Launch.py
MXNet provides a script tools/launch.py to make it easy to launch distributed training on a cluster with `ssh`, `mpi`, `sge` or `yarn`. If you don't already have a cluster, an easy way to set up a cluster of EC2 instances for distributed deep learning is using an [AWS CloudFormation template](https://github.com/awslabs/deeplearning-cfn).

Let's start with an example on how to use this. Assume we are in mxnet/example/image-classification and want to train LeNet to classify MNIST images with the script train_mnist.py.
On a single machine we can run the following to use 2 GPUs on a single machine with batch size of 128 on each GPU. Note that these command line arguments are specific to this script, and are not generic to any MXNet training script.

```
python train_mnist.py --network lenet --batch-size 256 --gpus 0,1 --kv-store device
```

Now suppose we have 4 machines which are ssh-able, meaning that they can ssh into each other without a password. We need to put the IPs of these machines into a file say `hosts`. For example the contents would be something like
 ```
 $ cat hosts
 172.30.0.172
 172.31.0.173
 172.30.1.174
 172.32.0.174
 ```

If the mxnet directory which contains launch.py and train_mnist.py

## Environment variables
 - MXNET_KVSTORE_REDUCTION_NTHREADS
Values: Int (default=4)
The number of CPU threads used for summing up big arrays.

- MXNET_KVSTORE_BIGARRAY_BOUND
Values: Int (default=1000000)
The minimum size of a “big array”.
When the array size is bigger than this threshold, MXNET_KVSTORE_REDUCTION_NTHREADS threads are used for reduction.
This parameter is also used as a load balancer in kvstore. It controls when to partition a single weight to all the servers. If the size of a single weight is less than MXNET_KVSTORE_BIGARRAY_BOUND then, it is sent to a single randomly picked server otherwise it is partitioned to all the servers.

- MXNET_ENABLE_GPU_P2P
Values: 0(false) or 1(true) (default=1)
If true, MXNet tries to use GPU peer-to-peer communication, if available on your device, when kvstore’s type is device. This is used only when kvstore has the type `device` in it.
