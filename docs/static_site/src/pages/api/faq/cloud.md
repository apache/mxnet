---
layout: page_category
title:  MXNet on the Cloud
category: faq
faq_c: Deployment Environments
question: How to run MXNet on AWS?
permalink: /api/faq/cloud
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

# MXNet on the Cloud

Deep learning can require extremely powerful hardware, often for unpredictable durations of time.
Moreover, _MXNet_ can benefit from both multiple GPUs and multiple machines.
Accordingly, cloud computing, as offered by AWS and others,
is especially well suited to training deep learning models.
Using AWS, we can rapidly fire up multiple machines with multiple GPUs each at will
and maintain the resources for precisely the amount of time needed.

## Set Up an AWS GPU Cluster from Scratch

In this document, we provide a step-by-step guide that will teach you
how to set up an AWS cluster with _MXNet_. We show how to:

- [Use Pre-installed EC2 GPU Instance](#use-pre-installed-ec2-gpu-instance)
- [Build and run MXNet on a single computer](#build-and-run-mxnet-on-a-gpu-instance)
- [Set up an EC2 GPU cluster for distributed training](#set-up-an-ec2-gpu-cluster-for-distributed-training)

### Use Pre-installed EC2 GPU Instance
The [Deep Learning AMIs](https://aws.amazon.com/marketplace/search/results?x=0&y=0&searchTerms=Deep+Learning+AMI)
are a series of images supported and maintained by Amazon Web Services for use
on Amazon Elastic Compute Cloud (Amazon EC2) and contain the latest MXNet release.

Now you can launch _MXNet_ directly on an EC2 GPU instance.
You can also use [Jupyter](https://jupyter.org) notebook on EC2 machine.
Here is a [good tutorial](https://github.com/dmlc/mxnet-notebooks)
on how to connect to a Jupyter notebook running on an EC2 instance.

### Set Up an EC2 GPU Instance from Scratch

[Deep Learning Base AMIs](https://aws.amazon.com/marketplace/search/results?x=0&y=0&searchTerms=Deep+Learning+Base+AMI)
provide a foundational image with NVIDIA CUDA, cuDNN, GPU drivers, Intel
MKL-DNN, Docker and Nvidia-Docker, etc. for deploying your own custom deep
learning environment. You may follow the [MXNet Build From Source
instructions](https://mxnet.apache.org/get_started/build_from_source) easily on
the Deep Learning Base AMIs.

### Set Up an EC2 GPU Cluster for Distributed Training

A cluster consists of multiple computers.
You can use one computer with _MXNet_ installed as the root computer for submitting jobs,and then launch several
slave computers to run the jobs. For example, launch multiple instances using an
AMI with dependencies installed. There are two options:

- Make all slaves' ports accessible (same for the root) by setting type: All TCP,
   Source: Anywhere in Configure Security Group.

- Use the same `pem` as the root computer to access all slave computers, and
   then copy the `pem` file into the root computer's `~/.ssh/id_rsa`. If you do this, all slave computers can be accessed with SSH from the root.

Now, run the CNN on multiple computers. Assume that we are on a working
directory of the root computer, such as `~/train`, and MXNet is built as `~/mxnet`.

1. Pack the _MXNet_ Python library into this working directory for easy
  synchronization:

  ```bash
  cp -r ~/mxnet/python/mxnet .
  cp ~/mxnet/lib/libmxnet.so mxnet/
  ```

  And then copy the training program:

  ```bash
  cp ~/mxnet/example/image-classification/*.py .
  cp -r ~/mxnet/example/image-classification/common .
  ```

2. Prepare a host file with all slaves private IPs. For example, `cat hosts`:

  ```bash
  172.30.0.172
  172.30.0.171
  ```

3. Assuming that there are two computers, train the CNN using two workers:

  ```bash
  ../../tools/launch.py -n 2 -H hosts --sync-dir /tmp/mxnet python train_mnist.py --kv-store dist_sync
  ```

***Note:*** Sometimes the jobs linger at the slave computers even though you've pressed `Ctrl-c`
at the root node. To terminate them, use the following command:

```bash
cat hosts | xargs -I{} ssh -o StrictHostKeyChecking=no {} 'uname -a; pgrep python | xargs kill -9'
```

***Note:*** The preceding example is very simple to train and therefore isn't a good
benchmark for distributed training. Consider using other [examples](https://github.com/dmlc/mxnet/tree/master/example/image-classification).

### More Options
#### Use Multiple Data Shards
It is common to pack a dataset into multiple files, especially when working in a distributed environment.
_MXNet_ supports direct loading from multiple data shards.
Put all of the record files into a folder, and point the data path to the folder.

#### Use YARN and SGE
Although using SSH can be simple when you don't have a cluster scheduling framework,
_MXNet_ is designed to be portable to various platforms.
We provide scripts available in [tracker](https://github.com/dmlc/dmlc-core/tree/master/tracker)
to allow running on other cluster frameworks, including Hadoop (YARN) and SGE.
We welcome contributions from the community of examples of running _MXNet_ on your favorite distributed platform.
