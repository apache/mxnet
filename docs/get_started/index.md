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

Please choose the programming language of your choice for the rest of this document.

<div class="btn-group opt-group" role="group">
<button type="button" class="btn btn-default opt active">Python</button>
<button type="button" class="btn btn-default opt">R</button>
<button type="button" class="btn btn-default opt">Scala</button>
<button type="button" class="btn btn-default opt">Julia</button>
<button type="button" class="btn btn-default opt">Perl</button>
</div>
<script type="text/javascript" src='../../_static/js/options.js'></script>

## Installation

<div class="btn-group opt-group" role="group">
<button type="button" class="btn btn-default opt active">Pre-built Binaries</button>
<button type="button" class="btn btn-default opt">Docker</button>
<button type="button" class="btn btn-default opt">Cloud</button>
<button type="button" class="btn btn-default opt">Build From Source</button>
</div> <!-- opt-group -->

<div class="pre-built-binaries">

<div class="r scala julia perl">
Pre-built binaries will be available soon.
</div>

<div class="python">

Installing the pre-build python package requires a recent version of `pip`,
which, for example, can be installed by

```bash
wget https://bootstrap.pypa.io/get-pip.py && sudo python get-pip.py
```

<div class="btn-group opt-group" role="group">
<button type="button" class="btn btn-default opt active">Linux</button>
<button type="button" class="btn btn-default opt">macOS</button>
<button type="button" class="btn btn-default opt">Windows</button>
</div> <!-- opt-group -->

<div class="windows">

Will be available soon.

</div> <!-- windows -->

<div class="macos">

Install by:

```bash
pip install mxnet
```

</div> <!-- macos -->

<div class="linux">

Use one of following commands to install the desired release:

```bash
pip install mxnet       # CPU
pip install mxnet-mkl   # CPU with MKL-DNN acceleration
pip install mxnet-cu75  # GPU with CUDA 7.5
pip install mxnet-cu80  # GPU with CUDA 8.0
```

The CUDA versions require both [CUDA](https://developer.nvidia.com/cuda-toolkit)
  and [cuDNN](https://developer.nvidia.com/cudnn) are installed.

</div> <!-- linux -->

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

Refer to the [building from source document](./build_from_source.md) for details
on building MXNet from source codes for various platforms.

</div> <!-- build-from-source -->


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

scala> (arr * 2 + 1).toArray
res0: Array[Float] = Array(3.0, 3.0, 3.0, 3.0, 3.0, 3.0)
```

```r
> require(mxnet)
Loading required package: mxnet
> a <- mx.nd.ones(c(2,3))
> a * 2 + 1
     [,1] [,2] [,3]
[1,]    3    3    3
[2,]    3    3    3
```

```julia
julia> using MXNet

julia> a = mx.ones((2,3))
mx.NDArray{Float32}(2,3)

julia> Array{Float32}(a * 2 + 1)
2×3 Array{Float32,2}:
 3.0  3.0  3.0
 3.0  3.0  3.0
```

```perl
pdl> use AI::MXNet qw(mx)
pdl> $a = mx->nd->ones([2, 3], ctx => mx->gpu())
pdl> print (($a * 2 + 1)->aspdl)
[
 [3 3 3]
 [3 3 3]
]
```

MXNet also provides a symbolic programming interface:

```python
>>> a = mx.sym.var('a')  # it requires the latest mxnet
>>> b = a * 2 + 1  # b is a Symbol object
>>> c = b.eval(a=mx.nd.ones((2,3)))
>>> c[0].asnumpy()  # the list of outputs
array([[ 3.,  3.,  3.],
       [ 3.,  3.,  3.]], dtype=float32)
```

```perl
pdl> use AI::MXNet qw(mx)
pdl> $a = mx->sym->var('a')
pdl> $b = $a * 2 + 1
pdl> $c = $b->eval(args => { a => mx->nd->ones([2,3]) })
pdl> print @{$c}[0]->aspdl
[
 [3 3 3]
 [3 3 3]
]
```

Run the above codes in GPU in straightforward:

```python
>>> a = mx.nd.ones((2, 3), mx.gpu())  # create a on GPU 0, then the result a*2+1 will sit on GPU 0 as well
>>> c = b.eval(a=a, ctx=mx.gpu())  # feed a as the input to eval b, the result c will be also on GPU 0
```

```r
> a <- mx.nd.ones(c(2,3), mx.gpu())
```

```julia
julia> a = mx.ones((2,3), mx.gpu())
```

```perl
pdl> $a = mx->nd->ones([2,3], ctx => mx->gpu())
```
In additional, MXNet provides a large number of neural network layers and
training modules to facilitate developing deep learning algorithms.

```python
>>> data = mx.sym.var('data')
>>> fc1  = mx.sym.FullyConnected(data, num_hidden=128)
>>> act1 = mx.sym.Activation(fc1, act_type="relu")
>>> fc2  = mx.sym.FullyConnected(act1, num_hidden=10)
>>> loss  = mx.sym.SoftmaxOutput(fc2)
>>> mod = mx.mod.Module(loss)
>>> mod.fit(train_data, ctx=[mx.gpu(0), mx.gpu(1)]) # fit on the training data by using 2 GPUs
```

```perl
pdl> $data  = mx->sym->var('data')
pdl> $fc1   = mx->sym->FullyConnected($data, num_hidden=>128)
pdl> $act1  = mx->sym.Activation($fc1, act_type=>"relu")
pdl> $fc2   = mx->sym->FullyConnected($act1, num_hidden=>10)
pdl> $loss  = mx->sym->SoftmaxOutput($fc2)
pdl> $mod   = mx->mod->Module($loss)
pdl> $mod->fit($train_data, ctx=>[mx->gpu(0), mx->gpu(1)]) # fit on the training data by using 2 GPUs
```

## Next Steps

* [Tutorials](http://mxnet.io/tutorials/index.html)
* [How To](http://mxnet.io/how_to/index.html)
* [API Documents](http://mxnet.io/api/index.html)
