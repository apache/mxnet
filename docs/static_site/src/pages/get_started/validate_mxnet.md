---
layout: page
title: Validate MXNet
action: Get Started
action_url: /get_started
permalink: /get_started/validate_mxnet
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

# Validate Your MXNet Installation

- [Python](#python)
- [Python with GPU](#python-with-gpu)
- [Verify GPU training](#verify-gpu-training)
- [Virtualenv](#virtualenv)
- [Docker with CPU](#docker-with-cpu)
- [Docker with GPU](#docker-with-gpu)
- [Cloud](#cloud)
- [C++](#alternative-language-bindings)
- [Clojure](#clojure)
- [Julia](#julia)
- [Perl](#perl)
- [R](#r)
- [Scala](#scala)


## Python

Start the python terminal.

```bash
$ python
```

Run a short *MXNet* python program to create a 2X3 matrix of ones, multiply each element in the matrix by 2 followed by adding 1. We expect the output to be a 2X3 matrix with all elements being 3.

```python
>>> import mxnet as mx
>>> a = mx.nd.ones((2, 3))
>>> b = a * 2 + 1
>>> b.asnumpy()
array([[ 3.,  3.,  3.],
       [ 3.,  3.,  3.]], dtype=float32)
```


## Python with GPU

This is similar to the previous example, but this time we use *mx.gpu()*, to set *MXNet* context to be GPUs.

```python
>>> import mxnet as mx
>>> a = mx.nd.ones((2, 3), mx.gpu())
>>> b = a * 2 + 1
>>> b.asnumpy()
array([[ 3.,  3.,  3.],
       [ 3.,  3.,  3.]], dtype=float32)
```


## Verify GPU Training

From the MXNet root directory run: `python example/image-classification/train_mnist.py --network lenet --gpus 0` to test GPU training.


## Virtualenv

Activate the virtualenv environment created for *MXNet*.

```bash
$ source ~/mxnet/bin/activate
```

After activating the environment, you should see the prompt as below.

```bash
(mxnet)$
```

Start the python terminal.

```bash
$ python
```

Run the previous Python example.


## Docker with CPU

Launch a Docker container with `mxnet/python` image and run example *MXNet* python program on the terminal.

```bash
$ docker run -it mxnet/python bash # Use sudo if you skip Step 2 in the installation instruction

# Start a python terminal
root@4919c4f58cac:/# python
```

Run the previous Python example.


## Docker with GPU

Launch a NVIDIA Docker container with `mxnet/python:gpu` image and run example *MXNet* python program on the terminal.

```bash
$ nvidia-docker run -it mxnet/python:gpu bash # Use sudo if you skip Step 2 in the installation instruction

# Start a python terminal
root@4919c4f58cac:/# python
```

Run the previous Python example and run the previous GPU examples.


## Cloud

Login to the cloud instance you launched, with pre-installed *MXNet*, following the guide by corresponding cloud provider.

Start the python terminal.

```bash
$ python
```

Run the previous Python example, and for GPU instances run the previous GPU example.


## Alternative Language Bindings

### C++

Please contribute an example!


### Clojure

Please contribute an example!


### Julia

Please contribute an example!


### Perl

Start the pdl2 terminal.

```bash
$ pdl2
```

Run a short *MXNet* Perl program to create a 2X3 matrix of ones, multiply each element in the matrix by 2 followed by adding 1. We expect the output to be a 2X3 matrix with all elements being 3.

```perl
pdl> use AI::MXNet qw(mx)
pdl> $a = mx->nd->ones([2, 3])
pdl> $b = $a * 2 + 1
pdl> print $b->aspdl

[
 [3 3 3]
 [3 3 3]
]
```

### R

Run a short *MXNet* R program to create a 2X3 matrix of ones, multiply each element in the matrix by 2 followed by adding 1. We expect the output to be a 2X3 matrix with all elements being 3.

```r
library(mxnet)
a <- mx.nd.ones(c(2,3), ctx = mx.cpu())
b <- a * 2 + 1
b
```

You should see the following output:

```r
[,1] [,2] [,3]
[1,]    3    3    3
[2,]    3    3    3
```


#### R with GPU

This is similar to the previous example, but this time we use *mx.gpu()*, to set *MXNet* context to be GPUs.

```r
library(mxnet)
a <- mx.nd.ones(c(2,3), ctx = mx.gpu())
b <- a * 2 + 1
b
```

You should see the following output:

```r
[,1] [,2] [,3]
[1,]    3    3    3
[2,]    3    3    3
```


### Scala

Run the <a href="https://github.com/apache/incubator-mxnet/tree/master/scala-package/mxnet-demo">MXNet-Scala demo project</a> to validate your Maven package installation.
