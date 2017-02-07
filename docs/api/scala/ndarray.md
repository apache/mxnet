# NDArray API


The NDArray package (`mxnet.ndarray`) contains tensor operations similar to `numpy.ndarray`. The syntax is also similar, except for some additional calls for dealing with I/O and multiple devices.

Topics:

* [Create NDArray](#create-ndarray)
* [NDArray Operations](#ndarray-operations)
* [NDArray API Reference](http://mxnet.io/api/scala/docs/index.html#ml.dmlc.mxnet.NDArray)

## Create NDArray

Create `mxnet.ndarray` as follows:

```scala
    import ml.dmlc.mxnet._
    // all-zero array of dimension 100x50
    val a = NDArray.zeros(100, 50)
    // all-one array of dimension 256x32x128x1
    val b = NDArray.ones(256, 32, 128, 1)
    // Initialize array with contents.
    // You can specify dimensions of an array using the Shape parameter while creating array.
    val c = NDArray.array(Array(1, 2, 3, 4, 5, 6), shape = Shape(2, 3))
```
This is similar to the way you use `numpy`.
## NDArray Operations

We provide some basic ndarray operations, like arithmetic and slice operations.

### Arithmetic Operations

Input:
```scala
    import ml.dmlc.mxnet._
    val a = NDArray.zeros(100, 50)
    a.shape
    val b = NDArray.ones(100, 50)
    // c and d will be calculated in parallel here!
    val c = a + b
    val d = a - b
    // inplace operation, b's contents will be modified, but c and d won't be affected.
    b += d
```

Output:
```scala
    ml.dmlc.mxnet.Shape = (100,50)
```

### Multiplication/Division Operations

Input:
```scala
    import ml.dmlc.mxnet._
    //Multiplication
    val ndones = NDArray.ones(2, 1)
    val ndtwos = ndones * 2
    ndtwos.toArray
    (ndones * ndones).toArray
    (ndtwos * ndtwos).toArray
    ndtwos *= ndtwos // inplace
    ndtwos.toArray

    //Division
    val ndones = NDArray.ones(2, 1)
    val ndzeros = ndones - 1f
    val ndhalves = ndones / 2
    ndhalves.toArray
    (ndhalves / ndhalves).toArray
    (ndones / ndones).toArray
    (ndzeros / ndones).toArray
    ndhalves /= ndhalves
    ndhalves.toArray
```

Output:
```scala
    Array[Float] = Array(2.0, 2.0)
    Array[Float] = Array(1.0, 1.0)
    Array[Float] = Array(4.0, 4.0)
    Array[Float] = Array(4.0, 4.0)

    Array[Float] = Array(0.5, 0.5)
    Array[Float] = Array(1.0, 1.0)
    Array[Float] = Array(1.0, 1.0)
    Array[Float] = Array(0.0, 0.0)
    Array[Float] = Array(1.0, 1.0)
```

### Slice Operations

```scala
    import ml.dmlc.mxnet._
    val a = NDArray.array(Array(1f, 2f, 3f, 4f, 5f, 6f), shape = Shape(3, 2))
    val a1 = a.slice(1)   
    assert(a1.shape === Shape(1, 2))
    assert(a1.toArray === Array(3f, 4f))

    val a2 = arr.slice(1, 3)
    assert(a2.shape === Shape(2, 2))
    assert(a2.toArray === Array(3f, 4f, 5f, 6f))
```

### Dot Product

Input:
```scala
     import ml.dmlc.mxnet._
     val arr1 = NDArray.array(Array(1f, 2f), shape = Shape(1, 2))
     val arr2 = NDArray.array(Array(3f, 4f), shape = Shape(2, 1))   
     val res = NDArray.dot(arr1, arr2)
     res.shape
     res.toArray
```

Output:
```scala
    ml.dmlc.mxnet.Shape = (1,1)
    Array[Float] = Array(11.0)
```

### Save and Load NDArray

You can use MXNet functions to save and load a list or dictionary of NDArrays from file systems, as follows:

```scala
    import ml.dmlc.mxnet._
    val a = NDArray.zeros(100, 200)
    val b = NDArray.zeros(100, 200)
    // save list of NDArrays
    NDArray.save("/path/to/array/file", Array(a, b))
    // save dictionary of NDArrays to AWS S3
    NDArray.save("s3://path/to/s3/array", Map("A" -> a, "B" -> b))
    // save list of NDArrays to hdfs.
    NDArray.save("hdfs://path/to/hdfs/array", Array(a, b))
    val from_file = NDArray.load("/path/to/array/file")
    val from_s3 = NDArray.load("s3://path/to/s3/array")
    val from_hdfs = NDArray.load("hdfs://path/to/hdfs/array")
```
The good thing about using the `save` and `load` interface is that you can use the format across all `mxnet` language bindings. They also already support Amazon S3 and HDFS.

### Multi-Device Support

Device information is stored in the `mxnet.Context` structure. When creating NDArray in MXNet, you can use the context argument (the default is the CPU context) to create arrays on specific devices as follows:

Input:
```scala
    import ml.dmlc.mxnet._
    val cpu_a = NDArray.zeros(100, 200)
    cpu_a.context
    val ctx = Context.gpu(0)
    val gpu_b = NDArray.zeros(Shape(100, 200), ctx)
    gpu_b.context
```

Output:
```scala
    ml.dmlc.mxnet.Context = cpu(0)
    ml.dmlc.mxnet.Context = gpu(0)
```

Currently, we *do not* allow operations among arrays from different contexts. To manually enable this, use the `copyto` member function to copy the content to different devices, and continue computation:

Input:
```scala
    import ml.dmlc.mxnet._
    val x = NDArray.zeros(100, 200)
    val ctx = Context.gpu(0)
    val y = NDArray.zeros(Shape(100, 200), ctx)
    val z = x + y
    val cpu_y = NDArray.zeros(100, 200)
    y.copyto(cpu_y)
    val z = x + cpu_y
```

Output:
```scala
    mxnet.base.MXNetError: [13:29:12] src/ndarray/ndarray.cc:33:
    Check failed: lhs.ctx() == rhs.ctx() operands context mismatch
```

## Next Steps
* See [KVStore API](kvstore.md) for multi-GPU and multi-host distributed training.
