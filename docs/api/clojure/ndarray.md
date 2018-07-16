# NDArray API


The NDArray API contains tensor operations similar to `numpy.ndarray`. The syntax is also similar, except for some additional calls for dealing with I/O and multiple devices.

Topics:

* [Create NDArray](#create-ndarray)
* [NDArray Operations](#ndarray-operations)
* [NDArray API Reference](http://mxnet.incubator.apache.org/api/clojure/docs/org.apache.clojure-mxnet.ndarray.html)

To follow along with this documentation, you can use this namespace with the needed requires:

```clojure
(ns docs.ndarray
  (:require [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.context :as context]))
```


## Create NDArray

Create `mxnet.ndarray` as follows:

```clojure

(def a (ndarray/zeros [100 50])) ;;all zero arrray of dimension 100 x 50
(def b (ndarray/ones [256 32 128 1])) ;; all one array of dimension
(def c (ndarray/array [1 2 3 4 5 6] [2 3])) ;; array with contents of a shape 2 x 3
```

There are also ways to convert a NDArray to a vec or get the shape or the NDArray as an object or vec as follows:

```clojure
(ndarray/->vec c) ;=> [1.0 2.0 3.0 4.0 5.0 6.0]
(ndarray/shape c) ;=> #object[org.apache.mxnet.Shape 0x583c865 "(2,3)"]
(ndarray/shape-vec c) ;=> [2 3]
```


## NDArray Operations

There are some basic NDArray operations, like arithmetic and slice operations.

### Arithmetic Operations

```clojure
(def a (ndarray/ones [1 5]))
(def b (ndarray/ones [1 5]))
(-> (ndarray/+ a b) (ndarray/->vec)) ;=>  [2.0 2.0 2.0 2.0 2.0]

;; original ndarrays are unchanged
(ndarray/->vec a) ;=> [1.0 1.0 1.0 1.0 1.0]
(ndarray/->vec b) ;=> [1.0 1.0 1.0 1.0 1.0]

;;inplace operators
(ndarray/+= a b)
(ndarray/->vec a) ;=>  [2.0 2.0 2.0 2.0 2.0]
```

Other arithmetic operations are similar.


### Slice Operations

```clojure
(def a (ndarray/array [1 2 3 4 5 6] [3 2]))
(def a1 (ndarray/slice a 1))
(ndarray/shape-vec a1) ;=> [1 2]
(ndarray/->vec a1) ;=> [3.0 4.0]

(def a2 (ndarray/slice a 1 3))
(ndarray/shape-vec a2) ;=>[2 2]
(ndarray/->vec a2) ;=> [3.0 4.0 5.0 6.0]
```

### Dot Product

```clojure
(def arr1 (ndarray/array [1 2] [1 2]))
(def arr2 (ndarray/array [3 4] [2 1]))
(def res (ndarray/dot arr1 arr2))
(ndarray/shape-vec res) ;=> [1 1]
(ndarray/->vec res) ;=> [11.0]
```

### Save and Load NDArray

You can use MXNet functions to save and load a list or dictionary of NDArrays from file systems, as follows:

```clojure
(ndarray/save "filename" {"arr1" arr1 "arr2" arr2})
;; you can also do "s3://path" or "hdfs"

To load

```clojure
(def from-file (ndarray/load "filename"))
from-file
;=>{"arr1" #object["org.apache.mxnet.NDArray@43d85753"], "arr2" #object["org.apache.mxnet.NDArray@5c93def4"]}
```

The good thing about using the `save` and `load` interface is that you can use the format across all `mxnet` language bindings. They also already support Amazon S3 and HDFS.

### Multi-Device Support

Device information is stored in the `mxnet.Context` structure. When creating NDArray in MXNet, you can use the context argument (the default is the CPU context) to create arrays on specific devices as follows:

```clojure
(def cpu-a (ndarray/zeros [100 200]))
(ndarray/context cpu-a) ;=> #object[org.apache.mxnet.Context 0x3f376123 "cpu(0)"]

(def gpu-b (ndarray/zeros [100 200] {:ctx (context/gpu 0)})) ;; to use with gpu

```

## Next Steps
* See [KVStore API](kvstore.md) for multi-GPU and multi-host distributed training.
