# MXNet: A Scalable Deep Learning Framework

MXNet is an open-source deep learning framework that allows you to define,
train, and deploy deep neural networks on a wide array of devices, from cloud
infrastructure to mobile devices.  It is highly scalable, allowing for fast
model training, and supports a flexible programming model and multiple
languages. MXNet allows you to mix symbolic and imperative programming flavors
to maximize both efficiency and productivity.  MXNet is built on a dynamic
dependency scheduler that automatically parallelizes both symbolic and
imperative operations on the fly.  A graph optimization layer on top of that
makes symbolic execution fast and memory efficient. The MXNet library is
portable and lightweight, and it scales to multiple GPUs and multiple machines.

Please choice the programming language for the rest of this document.

<div class='text-center'>
<div class="btn-group opt-group" role="group">
<button type="button" class="btn btn-default opt active">Python</button>
<button type="button" class="btn btn-default opt">R</button>
<button type="button" class="btn btn-default opt">Scala</button>
<button type="button" class="btn btn-default opt">Julia</button>
<button type="button" class="btn btn-default opt">Perl</button>
</div>
</div>

## Quick Overview

MXNet provides an imperative *n*-dimensional array interface:

```python
>>> import mxnet as mx
>>> a = mx.nd.ones((2, 3))
>>> b = a * 2 + 1
>>> b.asnumpy()  # print b by converting to a numpy.ndarray object
array([[ 3.,  3.,  3.],
       [ 3.,  3.,  3.]], dtype=float32)
```

```scala
scala> import ml.dmlc.mxnet._
import ml.dmlc.mxnet._
scala> val arr = NDArray.ones(2, 3)
arr: ml.dmlc.mxnet.NDArray = ml.dmlc.mxnet.NDArray@f5e74790
scala> arr.shape
res0: ml.dmlc.mxnet.Shape = (2,3)
scala> (arr * 2).toArray
res2: Array[Float] = Array(2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
scala> (arr * 2).shape
res3: ml.dmlc.mxnet.Shape = (2,3)
```

```r
> require(mxnet)
Loading required package: mxnet
> a <- mx.nd.ones(c(2,3))
> a
     [,1] [,2] [,3]
[1,]    1    1    1
[2,]    1    1    1
> a + 1
     [,1] [,2] [,3]
[1,]    2    2    2
[2,]    2    2    2
```

```julia
julia> using MXNet
julia> a = mx.ones((2,3), mx.gpu())
mx.NDArray{Float32}(2,3)
julia> Array{Float32}(a * 2)
2×3 Array{Float32,2}:
 2.0  2.0  2.0
 2.0  2.0  2.0
```

```perl
pdl> use AI::MXNet qw(mx)
pdl> $a = mx->nd->ones([2, 3], ctx => mx->gpu())
pdl> print (($a * 2)->aspdl)
[
 [2 2 2]
 [2 2 2]
]
```

Running the workload on GPUs is straightforward.

```python
>>> a = mx.nd.ones((2, 3), mx.gpu(0))  # create a on GPU 0
>>> b = a * 2 + 1
>>> b  # b will sit on GPU 0 as well
<NDArray 2x3 @gpu(0)>
>>> b.asnumpy()
array([[ 3.,  3.,  3.],
       [ 3.,  3.,  3.]], dtype=float32)
```

MXNet also provides a symbolic programming interface (Note: the following codes
needs `MXNet>=0.10`)

```python
>>> a = mx.sym.var('a')
>>> b = a * 2 + 1
>>> b
<Symbol _plusscalar0>
>>> c = b.eval(a=mx.nd.ones((2,3)))
>>> c[0].asnumpy()  # the list of outputs
c[0].asnumpy()
array([[ 3.,  3.,  3.],
       [ 3.,  3.,  3.]], dtype=float32)
>>> d = b.eval(a=mx.nd.ones((2,3), mx.gpu(0)), ctx=mx.gpu(0))  # run on GPU 0
```

In additional, MXNet provides a large number of neural network layers and
training modules to facilitate developing deep learning algorithms. The
following codes train a multilayer perceptron:

```python
>>> data = mx.sym.var('data')
>>> fc1  = mx.sym.FullyConnected(data, num_hidden=128)
>>> act1 = mx.sym.Activation(fc1, act_type="relu")
>>> fc2  = mx.sym.FullyConnected(act1, num_hidden=10)
>>> loss  = mx.sym.SoftmaxOutput(fc2)
>>> mod = mx.mod.Module(loss)
>>> mod.fit(data_reader, ...)
```


## Setup MXNet

<div class="btn-group opt-group" role="group">
<button type="button" class="btn btn-default opt">Build From Source</button>
<button type="button" class="btn btn-default opt active">Pre-Build Binaries</button>
<button type="button" class="btn btn-default opt">Docker</button>
<button type="button" class="btn btn-default opt">Cloud</button>
</div> <!-- opt-group -->

<div class="pre-build-binaries">

<div class="python">

Installing the pre-build python package requires a recent version of `pip`,
which, for example, can be installed by

```bash
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
```

<h3>macOS</h3>

Install by:

```bash
pip install mxnet
```

<h3>Linux</h3>

Use one of following commands to install the desired release:

```bash
pip install mxnet       # CPU
pip install mxnet-mkl   # CPU with MKL-DNN acceleration
pip install mxnet-cu75  # GPU with CUDA 7.5
pip install mxnet-cu80  # GPU with CUDA 8.0
```

The CUDA versions requires both [CUDA](https://developer.nvidia.com/cuda-toolkit)
  and [cuDNN](https://developer.nvidia.com/cudnn) are installed.

</div> <!-- python -->

</div> <!-- pre-build-binaries -->

<div class="cloud">

AWS images with MXNet installed:

- [Deep Learning AMI for Ubuntu](https://aws.amazon.com/marketplace/pp/B06VSPXKDX)
- [Deep Learning AMI for Amazon Linux](https://aws.amazon.com/marketplace/pp/B01M0AXXQB)

</div> <!-- cloud -->

<div class="docker">

Pre-build docker images are available at [docker hub](https://hub.docker.com/r/mxnet/).

<div class="python">

```bash
docker pull mxnet/python
docker pull mxnet/python:gpu
```

</div> <!-- python -->

<div class="scala">

```bash
docker pull mxnet/scala
```

</div> <!-- scala -->

<div class="r">

```bash
docker pull mxnet/r-lang
docker pull mxnet/r-lang:gpu
```

</div> <!-- r -->

<div class="julia">

```bash
docker pull mxnet/julia
docker pull mxnet/julia:gpu
```

</div> <!-- julia -->

Refer to [docker/](../../docker/) for more details.

</div> <!-- docker -->

<div class="build-from-source">

Refer to [setup](./setup.md) for details on building MXNet from source codes for
various systems.

</div> <!-- build-from-source -->

<script type="text/javascript" src='../../_static/js/options.js'></script>

## Next Steps

* [Tutorials](http://mxnet.io/tutorials/index.html)
* [How To](http://mxnet.io/how_to/index.html)
* [API Documents](http://mxnet.io/api/index.html)
