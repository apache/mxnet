# NDArray API


The NDArray package (`mxnet.ndarray`) contains tensor operations similar to `numpy.ndarray`. The syntax is also similar, except for some additional calls for dealing with I/O and multiple devices.

Topics:

* [Create NDArray](#create-ndarray)
* [NDArray Operations](#ndarray-operations)
* [NDArray API Reference](http://mxnet.io/api/scala/docs/index.html#ml.dmlc.mxnet.NDArray)

## Create NDArray

Create `mxnet.ndarray` as follows:

```scala
    scala> import ml.dmlc.mxnet._
    scala> // all-zero array of dimension 100x50
    scala> val a = NDArray.zeros(100, 50)
    scala> // all-one array of dimension 256x32x128x1
    scala> val b = NDArray.ones(256, 32, 128, 1)
    scala> // initialize array with contents, you can specify dimensions of array using Shape parameter while creating array.
    scala> val c = NDArray.array(Array(1, 2, 3, 4, 5, 6), shape = Shape(2, 3))
```
This is similar to the way you use `numpy`.
## NDArray Operations

We provide some basic ndarray operations, like arithmetic and slice operations.

### Arithmetic Operations

```scala
    scala> import ml.dmlc.mxnet._
    scala> val a = NDArray.zeros(100, 50)
    scala> a.shape
    ml.dmlc.mxnet.Shape = (100,50)
    scala> val b = NDArray.ones(100, 50)
    scala> // c and d will be calculated in parallel here!
    scala> val c = a + b
    scala> val d = a - b
    scala> // inplace operation, b's contents will be modified, but c and d won't be affected.
    scala> b += d
```

### Multiplication/Division Operations

```scala
    scala> import ml.dmlc.mxnet._
    //Multiplication
    scala> val ndones = NDArray.ones(2, 1)
    scala> val ndtwos = ndones * 2
    scala> ndtwos.toArray
    Array[Float] = Array(2.0, 2.0)
    scala> (ndones * ndones).toArray
    Array[Float] = Array(1.0, 1.0)
    scala> (ndtwos * ndtwos).toArray
    Array[Float] = Array(4.0, 4.0)
    scala> ndtwos *= ndtwos // inplace
    scala> ndtwos.toArray
    Array[Float] = Array(4.0, 4.0)

    //Division
    scala> val ndones = NDArray.ones(2, 1)
    scala> val ndzeros = ndones - 1f
    scala> val ndhalves = ndones / 2
    scala> ndhalves.toArray
    Array[Float] = Array(0.5, 0.5)
    scala> (ndhalves / ndhalves).toArray
    Array[Float] = Array(1.0, 1.0)
    scala> (ndones / ndones).toArray
    Array[Float] = Array(1.0, 1.0)
    scala> (ndzeros / ndones).toArray
    Array[Float] = Array(0.0, 0.0)
    scala> ndhalves /= ndhalves
    scala> ndhalves.toArray
    Array[Float] = Array(1.0, 1.0)
```

### Slice Operations

```scala
    scala> import ml.dmlc.mxnet._
    scala> val a = NDArray.array(Array(1f, 2f, 3f, 4f, 5f, 6f), shape = Shape(3, 2))
    scala> val a1 = a.slice(1)   
    scala> assert(a1.shape === Shape(1, 2))
    scala> assert(a1.toArray === Array(3f, 4f))

    scala> val a2 = arr.slice(1, 3)
    scala> assert(a2.shape === Shape(2, 2))
    scala> assert(a2.toArray === Array(3f, 4f, 5f, 6f))
```

### Dot Product

```scala
    scala> import ml.dmlc.mxnet._
    scala> val arr1 = NDArray.array(Array(1f, 2f), shape = Shape(1, 2))
    scala> val arr2 = NDArray.array(Array(3f, 4f), shape = Shape(2, 1))   
    scala> val res = NDArray.dot(arr1, arr2)
    scala> res.shape
    ml.dmlc.mxnet.Shape = (1,1)
    scala> res.toArray
    Array[Float] = Array(11.0)
```

### Save and Load NDArray

You can use MXNet functions to save and load a list or dictionary of NDArrays from file systems, as follows:

```scala
    scala> import ml.dmlc.mxnet._
    scala> val a = NDArray.zeros(100, 200)
    scala> val b = NDArray.zeros(100, 200)
    scala> // save list of NDArrays
    scala> NDArray.save("/path/to/array/file", Array(a, b))
    scala> // save dictionary of NDArrays to AWS S3
    scala> NDArray.save("s3://path/to/s3/array", Map("A" -> a, "B" -> b))
    scala> // save list of NDArrays to hdfs.
    scala> NDArray.save("hdfs://path/to/hdfs/array", Array(a, b))
    scala> val from_file = NDArray.load("/path/to/array/file")
    scala> val from_s3 = NDArray.load("s3://path/to/s3/array")
    scala> val from_hdfs = NDArray.load("hdfs://path/to/hdfs/array")
```
The good thing about using the `save` and `load` interface is that you can use the format across all `mxnet` language bindings. They also already support Amazon S3 and HDFS.

### Multi-Device Support

Device information is stored in the `mxnet.Context` structure. When creating NDArray in MXNet, you can use the context argument (the default is the CPU context) to create arrays on specific devices as follows:

```scala
    scala> import ml.dmlc.mxnet._
    scala> val cpu_a = NDArray.zeros(100, 200)
    scala> cpu_a.context
    ml.dmlc.mxnet.Context = cpu(0)
    scala> val ctx = Context.gpu(0)
    scala> val gpu_b = NDArray.zeros(Shape(100, 200), ctx)
    scala> gpu_b.context
    ml.dmlc.mxnet.Context = gpu(0)
```

Currently, we *do not* allow operations among arrays from different contexts. To manually enable this, use the `copyto` member function to copy the content to different devices, and continue computation:

```scala
    scala> import ml.dmlc.mxnet._
    scala> val x = NDArray.zeros(100, 200)
    scala> val ctx = Context.gpu(0)
    scala> val y = NDArray.zeros(Shape(100, 200), ctx)
    scala> val z = x + y
    mxnet.base.MXNetError: [13:29:12] src/ndarray/ndarray.cc:33:
    Check failed: lhs.ctx() == rhs.ctx() operands context mismatch
    scala> val cpu_y = NDArray.zeros(100, 200)
    scala> y.copyto(cpu_y)
    scala> val z = x + cpu_y
```

## Next Steps
* See [KVStore API](kvstore.md) for multi-GPU and multi-host distributed training.
