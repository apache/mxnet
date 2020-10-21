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

# Optimizing Deep Learning Computation Graphs with TensorRT

NVIDIA's TensorRT is a deep learning library that has been shown to provide large speedups when used for network inference. MXNet 1.5.0 and later versions ship with experimental integrated support for TensorRT. This means MXNet users can now make use of this acceleration library to efficiently run their networks. In this tutorial we'll see how to install, enable and run TensorRT with MXNet.  We'll also give some insight into what is happening behind the scenes in MXNet to enable TensorRT graph execution.

## Installation and Prerequisites
To use MXNet with TensorRT integration you'll have to follow the MXNet build from source instructions, and have a few extra packages installed on your machine. First ensure that you are running Ubuntu 18.04, and that you have updated your video drivers, and you have installed CUDA 10.1 or newer.  You'll need a Pascal or newer generation NVIDIA GPU.  You'll also have to download and install TensorRT libraries [instructions here](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html).  Once you have these prerequisites installed you can follow the [recommended instructions for building MXNet for NVIDIA GPUs](https://mxnet.apache.org/get_started/build_from_source#recommended-for-Systems-with-NVIDIA-GPUs-and-Intel-CPUs) but include the additional CMake flag -DUSE_TENSORRT=1.

## Sample Models
### Resnet 18
TensorRT is an inference only library, so for the purposes of this tutorial we will be using a pre-trained network, in this case a Resnet 18.  Resnets are a computationally intensive model architecture that are often used as a backbone for various computer vision tasks. Resnets are also commonly used as a reference for benchmarking deep learning library performance.  In this section we'll use a pretrained Resnet 18 from the [Gluon Model Zoo](/api/python/docs/api/gluon/model_zoo/index.html) and compare its inference speed with TensorRT using MXNet with TensorRT integration turned off as a baseline.

## Model Initialization
```python
import mxnet as mx
from mxnet.gluon.model_zoo import vision
import time
import os

ctx=mx.gpu(0)

batch_shape = (1, 3, 224, 224)
x = mx.nd.zeros(batch_shape, ctx=ctx)

model = vision.resnet18_v2(pretrained=True, ctx=ctx)
model.hybridize(static_shape=True, static_alloc=True)

```
In our first section of code we import the modules needed to run MXNet, and to time our benchmark runs.  We then download a pretrained version of Resnet18. We hybridize (link to hybridization) it with static_alloc and static_shape to get the best performance.

## MXNet Baseline Performance
```python
# Warmup
for i in range(0, 1000):
	out = model(x)
	mx.nd.waitall()

# Timing
start = time.time()
for i in range(0, 10000):
	out = model(x)
	mx.nd.waitall()
print(time.time() - start)
```

For this experiment we are strictly interested in inference performance, so to simplify the benchmark we'll pass a tensor filled with zeros as an input. 
To help improve the accuracy of our benchmarks we run a small number of predictions as a warmup before running our timed loop. This will ensure various lazy operations, which do not represent real-world usage, have completed before we measure relative performance improvement. On a system with a V100 GPU, the time taken for our MXNet baseline is **19.5s** (512 samples/s).

## MXNet with TensorRT Integration Performance
```python
[...]

model.optimize_for(x, backend='TensorRT', static_alloc=True, static_shape=True)

[...]
```

Next we'll run the same model with TensorRT enabled, and see how the performance compares.

To use TensorRT optimization with the Gluon, we need to call optimize_for with the TensorRT backend and provide some input data that will be used to infer shape and types (any sample representing the inference data). TensorRT backend supports only static shape, so we need to set static_alloc and static_shape to True.

This will run the subgraph partitioning and replace TensorRT compatible subgraphs with TensorRT ops containing the TensorRT engines. It's ready to be used.

```python
# Warmup
for i in range(0, 1000):
	out = model(x)
	out[0].wait_to_read()

# Timing
start = time.time()
for i in range(0, 10000):
	out = model(x)
	out[0].wait_to_read()
print(time.time() - start)
```

We run timing with a warmup once again, and on the same machine, run in **12.7s** (787 samples/s). A 1.5x speed improvement!  Speed improvements when using libraries like TensorRT can come from a variety of optimizations, but in this case our speedups are coming from a technique known as [operator fusion](http://ziheng.org/2016/11/21/fusion-and-runtime-compilation-for-nnvm-and-tinyflow/).

## FP16

We can give a simple speed up by turning on TensorRT FP16. This optimization comes almost as a freebie and doesn't need any other use effort than adding the optimize_for parameter precision.

```python
[...]

model.optimize_for(x, backend='TensorRT', static_alloc=True, static_shape=True, backend_opts={'precision':'fp16'})

[...]
```

We run timing with a warmup once more and we get **7.8s** (1282 samples/s). That's 2.5x speedup compared to the default MXNet!
All the ops used in ResNet-18 are FP16 compatible, so the TensorRT engine was able to run FP16 kernels, hence the extra speed up.


## Operators and Subgraph Fusion

Behind the scenes a number of interesting things are happening to make these optimizations possible, and most revolve around subgraphs and operator fusion.  As we can see in the images below, neural networks can be represented as computation graphs of operators (nodes in the graphs).  Operators can perform a variety of functions, but most run simple mathematics and linear algebra on tensors.  Often these operators run more efficiently if fused together into a large CUDA kernel that is executed on the GPU in a single call.  What the MXNet TensorRT integration enables is the ability to scan the entire computation graph, identify interesting subgraphs and optimize them with TensorRT.

This means that when an MXNet computation graph is constructed, it will be parsed to determine if there are any sub-graphs that contain operator types that are supported by TensorRT.  If MXNet determines that there are one (or many) compatible subgraphs during the graph-parse, it will extract these graphs and replace them with special TensorRT nodes (visible in the diagrams below).  As the graph is executed, whenever a TensorRT node is reached the graph will make a library call to TensorRT.  TensorRT will then run its own implementation of the subgraph, potentially with many operators fused together into a single CUDA kernel.

During this process MXNet will take care of passing along the input to the node and fetching the results.  MXNet will also attempt to remove any duplicated weights (parameters) during the graph initialization to keep memory usage low.  That is, if there are graph weights that are used only in the TensorRT sections of the graph, they will be removed from the MXNet set of parameters, and their memory will be freed.

The examples below shows a Gluon implementation of a Wavenet before and after a TensorRT graph pass. You can see that for this network TensorRT supports a subset of the operators involved. This makes it an interesting example to visualize, as several subgraphs are extracted and replaced with special TensorRT nodes. The Resnet used as an example above would be less interesting to visualization. The entire Resnet graph is supported by TensorRT, and hence the optimized graph would be a single TensorRT node.  If your browser is unable to render svg files you can view the graphs in png format: [unoptimized](wavenet_unoptimized.svg) and [optimized](wavenet_optimized.svg).

## Before
![before](wavenet_unoptimized.svg)

## After
![after](wavenet_optimized.svg)

## Subgraph API
As of MXNet 1.5, MXNet developers have integrated TensorRT with MXNet via a Subgraph API.  Read more about the design of the API [here](https://cwiki.apache.org/confluence/display/MXNET/MXNet+Graph+Optimization+and+Quantization+based+on+subgraph+and+MKL-DNN).

## Thanks
Thanks to NVIDIA for contributing this feature, and specifically thanks to Marek Kolodziej and Clement Fuji-Tsang.  Thanks to Junyuan Xie and Jun Wu for the code reviews and design feedback, and to Aaron Markham for the copy review.