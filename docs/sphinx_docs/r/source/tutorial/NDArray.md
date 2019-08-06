# NDArray: Vectorized Tensor Computations on CPUs and GPUs

``NDArray`` is the basic vectorized operation unit in **mxnet** for vector, matrix and tensor computations. Users can perform usual calculations as on an R array, but with additional features including:

- Multiple devices: All operations can be run on various devices including CPUs and GPUs.

- Automatic parallelization: All operations are automatically executed in parallel with each other.

We first load the **mxnet** package. 
**Note:**   A few commands in this tutorial will produce errors unless you have properly installed the GPU-version of MXNet.  To avoid these issues later on, we preset the Boolean flag ``use_gpu`` based on whether or not a GPU is detected in the current environment.

```{.python .input  n=45}
require(mxnet)
use_gpu <- !inherits(try(mx.nd.zeros(1,mx.gpu()), silent = TRUE), 'try-error') # TRUE if GPU is detected.
```

## Create and Initialize

Let's create a ``NDArray`` on our CPU device. The below code will instantiate a 2-by-3 matrix where each entry = 0.  The CPU device is chosen by default for NDArrays, so we do not have to explicitly specify it.

```{.python .input  n=41}
a <- mx.nd.zeros(c(2, 3), mx.cpu())
b <- mx.nd.zeros(c(2, 3)) # another NDArray on CPU
```

We can also create a ``NDArray`` on our GPU device. 
Typically for CUDA-enabled devices, the device id of a GPU starts from 0. That’s why we pass in 0 to the GPU id in the command below.
If available, you can feel free to run the rest of the operations in this tutorial on your GPU by additionally specifying ``mx.gpu`` in the initialization of each ``NDArray``.

**Note**: The below command can be used to create a 2-by-3 matrix on the GPU with device id = 0, but will not work unless you have properly installed the GPU-version of MXNet.

```{.python .input  n=42}
if (use_gpu) {
    c <- mx.nd.zeros(c(2, 3), mx.gpu(0))
}
```

We can initialize a ``NDArray`` object in various ways:

```{.python .input  n=4}
a <- mx.nd.ones(c(4, 4)) # 4x4 matrix where each entry = 1
b <- mx.rnorm(c(4, 5))  # 4x5 matrix of random N(0,1) entries
c <- mx.nd.array(1:5)  # vector [1,2,3,4,5]
dr <- matrix(1:6,nrow=2) # R matrix
d <- mx.nd.array(dr) # NDArray containing same values as dr
d
```

To view the numbers in an NDArray or process the array using R functions, we can simply convert it back to an R array via ``as.array``:

```{.python .input  n=5}
NDarrayObj <- mx.nd.ones(c(1, 3))
RarrayObj <- as.array(NDarrayObj)
class(RarrayObj)
RarrayObj
```

## Performing Basic Operations

Just as with R arrays, you can perform elemental-wise operations on NDArray objects as follows:

```{.python .input  n=6}
a <- mx.nd.ones(c(2, 4)) * 2
as.array(a)
```

```{.python .input  n=8}
b <- mx.nd.ones(c(2, 4)) / 8
as.array(b)
```

```{.python .input  n=9}
c <- a + b
as.array(c)
```

```{.python .input  n=10}
d <- c / a - 5
as.array(d)
```

We can also perform more complex matrix operations on NDArrays, such as matrix multiplication:

```{.python .input  n=11}
e <- mx.nd.dot(mx.nd.transpose(a), b)
e
```

If two ``NDArray`` objects are located on different devices, we need to explicitly move them to the same device before performing an operation that involves them both.

**Note**: The below commands can be used to mix GPU and CPU operations, but will not work unless you have properly installed the GPU-version of MXNet.

```{.python .input  n=43}
if (use_gpu) {
    a <- mx.nd.ones(c(2, 3)) * 2
    b <- mx.nd.ones(c(2, 3), mx.gpu()) / 8
    c <- mx.nd.copyto(a, mx.gpu()) * b
}
```

##  Load and Save

You can save a list of ``NDArray`` objects to a file on your disk:

```{.python .input  n=12}
a <- mx.nd.ones(c(2, 3))
b <- mx.nd.zeros(c(1,2))
mx.nd.save(list(a,b), "temp.ndarrays")
```

And load these objects into R from a saved file:

```{.python .input  n=13}
a <- 1; b <- 1
ndlist <- mx.nd.load("temp.ndarrays")
a <- ndlist[[1]]
b <- ndlist[[2]]
as.array(a)
as.array(b)
```

Alternatively, we can directly save data to and load it from a distributed file system, such as Amazon S3 and HDFS, by issuing commands that look like:

```{.python .input  n=26}
if (FALSE) { # this code will produce error (need to properly specify bucket/user name)
    mx.nd.save(list(a), "s3://mybucket/mydata.bin")
    mx.nd.save(list(a), "hdfs///users/myname/mydata.bin")
}
```

## Automatic Parallelization

NDArray can automatically execute operations in parallel, which is useful when using multiple resources such as: CPU cards, GPU cards, and CPU-to-GPU memory bandwidth.

For example, if we write ``a <- a + 1`` followed by ``b <- b + 1``, where ``a`` is on a CPU and ``b`` is on a GPU, executing them in parallel improves efficiency. Because copying data between CPUs and GPUs is expensive, running in parallel with other computations further increases efficiency.

```{.python .input  n=22}
a <- mx.nd.ones(c(2,3))
b <- a
c <- mx.nd.copyto(a, mx.cpu())
a <- a + 1
b <- b * 3
c <- c * 3
```

It’s hard to find the code that can be executed in parallel by eye. In the above example, ``a <- a + 1`` and ``c <- c * 3`` can be executed in parallel, but ``a <- a + 1`` and ``b <- b * 3`` should be in sequential.

Luckily, MXNet can automatically resolve the dependencies and execute operations in parallel accurately. This allows us to write our program assuming there is only a single thread. MXNet will automatically dispatch the program to multiple devices.

MXNet achieves this with lazy evaluation. Each operation is issued to an internal engine, and then returned. For example, if we run ``a <- a + 1``, it returns immediately after pushing the plus operator to the engine. This asynchronous processing allows us to push more operators to the engine. It determines the read and write dependencies and the best way to execute them in parallel.

The actual computations are finished, allowing us to copy the results someplace else, such as ``as.array(a)`` or ``mx.nd.save(a, "temp.dat")``. To write code that can take advantage of the highest degree of parallelization, we should postpone when we convert the results out of ``NDArray`` format until they are actually needed in another format.  For the above example, we should not call ``mx.nd.save`` on ``a`` or ``b`` until after the NDArray-operation involving ``c`` has been declared, in order to ensure maximal efficiency.
