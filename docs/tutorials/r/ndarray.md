# NDArray: Vectorized tensor computations on CPUs and GPUs

`NDArray` is the basic vectorized operation unit in MXNet for matrix and tensor computations.
Users can perform usual calculations as on R"s array, but with two additional features:

1.  **multiple devices**: all operations can be run on various devices including
CPU and GPU
2. **automatic parallelization**: all operations are automatically executed in
   parallel with each other

## Create and Initialization

Let"s create `NDArray` on either GPU or CPU


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
# c <- mx.nd.zeros(c(2, 3), mx.gpu(0)) # create a 2-by-3 matrix on gpu 0, if you have CUA enabled.
```

As a side note, normally for CUDA enabled devices, the device id of GPU starts from 0.
So that is why we passed in 0 to GPU id. We can also initialize an `NDArray` object in various ways:


```r
a <- mx.nd.ones(c(4, 4))
b <- mx.rnorm(c(4, 5))
c <- mx.nd.array(1:5)
```

To check the numbers in an `NDArray`, we can simply run


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

## Basic Operations

### Elemental-wise operations

You can perform elemental-wise operations on `NDArray` objects:


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

If two `NDArray`s sit on different divices, we need to explicitly move them
into the same one. For instance:


```r
a <- mx.nd.ones(c(2, 3)) * 2
b <- mx.nd.ones(c(2, 3), mx.gpu()) / 8
c <- mx.nd.copyto(a, mx.gpu()) * b
as.array(c)
```

### Load and Save

You can save a list of `NDArray` object to your disk with `mx.nd.save`:


```r
a <- mx.nd.ones(c(2, 3))
mx.nd.save(list(a), "temp.ndarray")
```

You can also load it back easily:


```r
a <- mx.nd.load("temp.ndarray")
as.array(a[[1]])
```

```
##      [,1] [,2] [,3]
## [1,]    1    1    1
## [2,]    1    1    1
```

In case you want to save data to the distributed file system such as S3 and HDFS,
we can directly save to and load from them. For example:


```r
mx.nd.save(list(a), "s3://mybucket/mydata.bin")
mx.nd.save(list(a), "hdfs///users/myname/mydata.bin")
```

## Automatic Parallelization

`NDArray` can automatically execute operations in parallel. It is desirable when we
use multiple resources such as CPU, GPU cards, and CPU-to-GPU memory bandwidth.

For example, if we write `a <- a + 1` followed by `b <- b + 1`, and `a` is on CPU while
`b` is on GPU, then want to execute them in parallel to improve the
efficiency. Furthermore, data copy between CPU and GPU are also expensive, we
hope to run it parallel with other computations as well.

However, finding the codes can be executed in parallel by eye is hard. In the
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
execute operations in parallel with correctness guaranteed. In other words, we
can write program as by assuming there is only a single thread, while MXNet will
automatically dispatch it into multi-devices, such as multi GPU cards or multi
machines.

It is achieved by lazy evaluation. Any operation we write down is issued into a
internal engine, and then returned. For example, if we run `a <- a + 1`, it
returns immediately after pushing the plus operator to the engine. This
asynchronous allows us to push more operators to the engine, so it can determine
the read and write dependency and find a best way to execute them in
parallel.

The actual computations are finished if we want to copy the results into some
other place, such as `as.array(a)` or `mx.nd.save(a, "temp.dat")`. Therefore, if we
want to write highly parallelized codes, we only need to postpone when we need
the results.

# Recommended Next Steps
* [Symbol](http://mxnet.io/tutorials/r/symbol.html)
* [Write and use callback functions](http://mxnet.io/tutorials/r/CallbackFunctionTutorial.html)
* [Neural Networks with MXNet in Five Minutes](http://mxnet.io/tutorials/r/fiveMinutesNeuralNetwork.html)
* [Classify Real-World Images with Pre-trained Model](http://mxnet.io/tutorials/r/classifyRealImageWithPretrainedModel.html)
* [Handwritten Digits Classification Competition](http://mxnet.io/tutorials/r/mnistCompetition.html)
* [Character Language Model using RNN](http://mxnet.io/tutorials/r/charRnnModel.html)