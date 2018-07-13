# MXNet Clojure Symbolic API

Topics:

* [How to Compose Symbols](#how-to-compose-symbols)
* [More Complicated Compositions](#more-complicated-compositions)
* [Group Multiple Symbols](#group-multiple-symbols)
* [Serialization](#serialization)
* [Executing Symbols](#executing-symbols)
* [Multiple Outputs](#multiple-outputs)
* [Symbol API Reference](http://mxnet.incubator.apache.org/api/clojure/docs/org.apache.clojure-mxnet.symbol.html)


We also highly encourage you to read [Symbolic Configuration and Execution in Pictures](symbol_in_pictures.md).

To follow along with this documentation, you can use this namespace to with the following requirements:

```clojure
(ns docs.symbol
  (:require [org.apache.clojure-mxnet.executor :as executor]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.context :as context]))
```


## How to Compose Symbols

The Symbolic API provides a way to configure computation graphs.
You can configure the graphs either at the level of neural network layer operations or as fine-grained operations.

The following example configures a two-layer neural network.

```clojure
(def data (sym/variable "data"))
(def fc1 (sym/fully-connected "fc1" {:data data :num-hidden 128}))
(def act1 (sym/activation "act1" {:data fc1 :act-type "relu"}))
(def fc2 (sym/fully-connected "fc2" {:data act1 :num-hidden 64}))
(def net (sym/softmax-output "out" {:data fc2}))
```

This can also be combined more dynamically with the `as->` Clojure threading form.

```clojure
(as-> (sym/variable "data") data
  (sym/fully-connected "fc1" {:data data :num-hidden 128})
  (sym/activation "act1" {:data data :act-type "relu"})
  (sym/fully-connected "fc2" {:data data :num-hidden 64})
  (sym/softmax-output "out" {:data data}))

net ;=> #object[org.apache.mxnet.Symbol 0x5c78c8c2 "org.apache.mxnet.Symbol@5c78c8c2"] 
```

The basic arithmetic operators (plus, minus, div, multiplication) work as expected.

The following example creates a computation graph that adds two inputs together.

```clojure
(def a (sym/variable "a"))
(def b (sym/variable "b"))
(def c (sym/+ a b))
```

## More Complicated Compositions

MXNet provides well-optimized symbols for layers commonly used in deep learning (see src/operator). We can also define new operators in Python. The following example first performs an element-wise add between two symbols, then feeds them to the fully connected operator:

```clojure
(def lhs (sym/variable "data1"))
(def rhs (sym/variable "data2"))
(def net (sym/fully-connected "fc1" {:data (sym/+ lhs rhs) :num-hidden 128}))
(sym/list-arguments net) ;=> ["data1" "data2" "fc1_weight" "fc1_bias"]
```

## Group Multiple Symbols

To construct neural networks with multiple loss layers, we can use `group` to group multiple symbols together. The following example groups two outputs:

```clojure
(def net (sym/variable "data"))
(def fc1 (sym/fully-connected {:data net :num-hidden 128}))
(def net2 (sym/activation {:data fc1 :act-type "relu"}))
(def out1 (sym/softmax-output {:data net2}))
(def out2 (sym/linear-regression-output {:data net2}))
(def group (sym/group [out1 out2]))
(sym/list-outputs group)
;=> ["softmaxoutput0_output" "linearregressionoutput0_output"]
```

## Serialization
You can use the [`save`](docs/org.apache.clojure-mxnet.symbol.html#var-save) and [`load`](docs/org.apache.clojure-mxnet.symbol.html#var-load) functions to serialize the Symbol objects. The advantage of using save and load functions is that it is language agnostic and cloud friendly. The symbol is saved in JSON format. You can also get a JSON string directly using mxnet.Symbol.toJson. Refer to API documentation for more details.

 The following example shows how to save a symbol to a file, load it back, and compare two symbols using a JSON string. You can also save to S3 as well

```clojure
(def a (sym/variable "a"))
(def b (sym/variable "b"))
(def c (sym/+ a b))
(sym/save c "symbol-c.json")
(def c2 (sym/load "symbol-c.json"))
(= (sym/to-json c) (sym/to-json c2)) ;=>true
```


## Executing Symbols

To execute symbols, first we need to define the data that they should run on. We can do it by using the bind method, which accepts device context and a dict mapping free variable names to NDArrays as arguments and returns an executor. The executor provides forward method for evaluation and an attribute outputs to get all the results.

```clojure
(def a (sym/variable "a"))
(def b (sym/variable "b"))
(def c (sym/+ a b))

(def ex (sym/bind c {"a" (ndarray/ones [2 2]) "b" (ndarray/ones [2 2])}))
(-> (executor/forward ex)
    (executor/outputs)
    (first)
    (ndarray/->vec));=>  [2.0 2.0 2.0 2.0]
```

We can evaluate the same symbol on GPU with different data.
_To do this you must have the correct native library jar defined as a dependency_

**Note In order to execute the following section on a cpu set gpu_device to (cpu)**


```clojure
(def ex (sym/bind c (context/gpu 0) {"a" (ndarray/ones [2 2]) "b" (ndarray/ones [2 2])}))
```

## Multiple Outputs

To construct neural networks with multiple loss layers, we can use mxnet.sym.Group to group multiple symbols together. The following example groups two outputs:

```clojure
(def net (sym/variable "data"))
(def fc1 (sym/fully-connected {:data net :num-hidden 128}))
(def net2 (sym/activation {:data fc1 :act-type "relu"}))
(def out1 (sym/softmax-output {:data net2}))
(def out2 (sym/linear-regression-output {:data net2}))
(def group (sym/group [out1 out2]))
(sym/list-outputs group);=> ["softmaxoutput0_output" "linearregressionoutput0_output"]
```

After you get the ```group```, you can bind on ```group``` instead.
The resulting executor will have two outputs, one for `linerarregressionoutput_output` and one for `softmax_output`.

## Next Steps
* See [NDArray API](ndarray.md) for vector/matrix/tensor operations.
* See [KVStore API](kvstore.md) for multi-GPU and multi-host distributed training.
