---
layout: page_api
title: NDArray
is_tutorial: true
tag: r
permalink: /api/r/docs/tutorials/ndarray
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


# NDArray: Vectorized Tensor Computations on CPUs and GPUs

`NDArray` is the basic vectorized operation unit in MXNet for matrix and tensor computations.
Users can perform usual calculations as on an R"s array, but with two additional features:



- Multiple devices: All operations can be run on various devices including
CPUs and GPUs.


- Automatic parallelization: All operations are automatically executed in
   parallel with each other.

## Create and Initialize

Let"s create `NDArray` on either a GPU or a CPU:


```r
require(mxnet)
```

```
## Loading required package: mxnet
## Loading required package: methods
```

```r
a <- mx.nd.zeros(c(2, 3)) # create a 2-by-3 matrix on cpu
b <- mx.nd.zeros(c(2, 3), mx.cpu()) # create a 2-by-3 matrix on cpu
# c <- mx.nd.zeros(c(2, 3), mx.gpu(0)) # create a 2-by-3 matrix on gpu 0, if you have CUDA enabled.
```

Typically for CUDA-enabled devices, the device id of a GPU starts from 0.
That's why we passed in 0 to the GPU id.

We can initialize an `NDArray` object in various ways:


```r
a <- mx.nd.ones(c(4, 4))
b <- mx.rnorm(c(4, 5))
c <- mx.nd.array(1:5)
```

To check the numbers in an `NDArray`, we can simply run:


```r
a <- mx.nd.ones(c(2, 3))
b <- as.array(a)
class(b)
```

```
## [1] "matrix"
```

```r
b
```

```
##      [,1] [,2] [,3]
## [1,]    1    1    1
## [2,]    1    1    1
```

## Performing Basic Operations

### Elemental-wise Operations

You can perform elemental-wise operations on `NDArray` objects, as follows:


```r
a <- mx.nd.ones(c(2, 4)) * 2
b <- mx.nd.ones(c(2, 4)) / 8
as.array(a)
```

```
##      [,1] [,2] [,3] [,4]
## [1,]    2    2    2    2
## [2,]    2    2    2    2
```

```r
as.array(b)
```

```
##       [,1]  [,2]  [,3]  [,4]
## [1,] 0.125 0.125 0.125 0.125
## [2,] 0.125 0.125 0.125 0.125
```

```r
c <- a + b
as.array(c)
```

```
##       [,1]  [,2]  [,3]  [,4]
## [1,] 2.125 2.125 2.125 2.125
## [2,] 2.125 2.125 2.125 2.125
```

```r
d <- c / a - 5
as.array(d)
```

```
##         [,1]    [,2]    [,3]    [,4]
## [1,] -3.9375 -3.9375 -3.9375 -3.9375
## [2,] -3.9375 -3.9375 -3.9375 -3.9375
```

If two `NDArray`s are located on different devices, we need to explicitly move them to the same one. For instance:


```r
a <- mx.nd.ones(c(2, 3)) * 2
b <- mx.nd.ones(c(2, 3), mx.gpu()) / 8
c <- mx.nd.copyto(a, mx.gpu()) * b
as.array(c)
```

### Loading and Saving

You can save a list of `NDArray` object to your disk with `mx.nd.save`:


```r
a <- mx.nd.ones(c(2, 3))
mx.nd.save(list(a), "temp.ndarray")
```

You can load it back easily:


```r
a <- mx.nd.load("temp.ndarray")
as.array(a[[1]])
```

```
##      [,1] [,2] [,3]
## [1,]    1    1    1
## [2,]    1    1    1
```

We can directly save data to and load it from a distributed file system, such as Amazon S3 and HDFS:


```r
mx.nd.save(list(a), "s3://mybucket/mydata.bin")
mx.nd.save(list(a), "hdfs///users/myname/mydata.bin")
```

## Automatic Parallelization

`NDArray` can automatically execute operations in parallel. Automatic parallelization is useful when
using multiple resources, such as CPU cards, GPU cards, and CPU-to-GPU memory bandwidth.

For example, if we write `a <- a + 1` followed by `b <- b + 1`, and `a` is on a CPU and
`b` is on a GPU, executing them in parallel improves
efficiency. Furthermore, because copying data between CPUs and GPUs are also expensive, running in parallel with other computations further increases efficiency.

It's hard to find the code that can be executed in parallel by eye. In the
following example, `a <- a + 1` and `c <- c * 3` can be executed in parallel, but `a <- a + 1` and
`b <- b * 3` should be in sequential.


```r
a <- mx.nd.ones(c(2,3))
b <- a
c <- mx.nd.copyto(a, mx.cpu())
a <- a + 1
b <- b * 3
c <- c * 3
```

Luckily, MXNet can automatically resolve the dependencies and
execute operations in parallel accurately. This allows us to write our program assuming there is only a single thread. MXNet will
automatically dispatch the program to multiple devices.

MXNet achieves this with lazy evaluation. Each operation is issued to an
internal engine, and then returned. For example, if we run `a <- a + 1`, it
returns immediately after pushing the plus operator to the engine. This
asynchronous processing allows us to push more operators to the engine. It determines
the read and write dependencies and the best way to execute them in
parallel.

The actual computations are finished, allowing us to copy the results someplace else, such as `as.array(a)` or `mx.nd.save(a, "temp.dat")`. To write highly parallelized codes, we only need to postpone when we need
the results.

## Next Steps
* [Symbol](/api/r/docs/tutorials/symbol)
* [Classify Real-World Images with Pre-trained Model](/api/r/docs/tutorials/classify_real_image_with_pretrained_model)
* [Character Language Model using RNN](/api/r/docs/tutorials/char_rnn_model)
