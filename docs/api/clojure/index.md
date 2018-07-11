# MXNet - Clojure API
MXNet supports the Clojure programming language. The MXNet Clojure package brings flexible and efficient GPU
computing and state-of-art deep learning to Clojure. It enables you to write seamless tensor/matrix computation with multiple GPUs in Clojure. It also lets you construct and customize the state-of-art deep learning models in Clojure, and apply them to tasks, such as image classification and data science challenges.

See the [MXNet Clojure API Documentation](docs/index.html) for detailed API information.


## Tensor and Matrix Computations
You can perform tensor or matrix computation in pure Clojure:

```clojure
(def arr (ndarray/ones [2 3]))

arr ;=> #object[org.apache.mxnet.NDArray 0x597d72e "org.apache.mxnet.NDArray@e35c3ba9"]

(ndarray/shape-vec arr) ;=>  [2 3]

(-> (ndarray/* arr 2)
    (ndarray/->vec)) ;=> [2.0 2.0 2.0 2.0 2.0 2.0]

(ndarray/shape-vec (ndarray/* arr 2)) ;=> [2 3]

```

## Clojure API Tutorials
* [Module API is a flexible high-level interface for training neural networks.](module.html)
* [Symbolic API performs operations on NDArrays to assemble neural networks from layers.](symbol.html)
* [NDArray API performs vector/matrix/tensor operations.](ndarray.html)
* [KVStore API performs multi-GPU and multi-host distributed training.](kvstore.html)


## Related Resources
* [MXNet Clojure API Documentation](docs/index.html)
