---
layout: page_api
title: NDArray
is_tutorial: true
tag: r
permalink: /api/r/docs/tutorials/symbol
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

# Symbol and Automatic Differentiation

The computational unit `NDArray` requires a way to construct neural networks. MXNet provides a symbolic interface, named Symbol, to do this. Symbol combines both flexibility and efficiency.

## Basic Composition of Symbols

The following code creates a two-layer perceptron network:

```
require(mxnet)
## [1] "Rcpp_MXSymbol"
## attr(,"package")
## [1] "mxnet"
```

Each symbol takes a (unique) string name. *Variable* often defines the inputs,
or free variables. Other symbols take a symbol as the input (*data*),
and may accept other hyper parameters, such as the number of hidden neurons (*num_hidden*)
or the activation type (*act_type*).

We can also specify the names explicitly:

```r
data <- mx.symbol.Variable("data")
w <- mx.symbol.Variable("myweight")
net <- mx.symbol.FullyConnected(data=data, weight=w, name="fc1", num_hidden=128)
arguments(net)
```

```
## [1] "data"     "myweight" "fc1_bias"
```

## More Complicated Composition of Symbols

MXNet provides well-optimized symbols for
commonly used layers in deep learning. You can also define new operators
in Python. The following example first performs an element-wise add between two
symbols, then feeds them to the fully connected operator:


```r
lhs <- mx.symbol.Variable("data1")
rhs <- mx.symbol.Variable("data2")
net <- mx.symbol.FullyConnected(data=lhs + rhs, name="fc1", num_hidden=128)
arguments(net)
```

```
## [1] "data1"      "data2"      "fc1_weight" "fc1_bias"
```

We can construct a symbol more flexibly than by using the single
forward composition, for example:


```r
net <- mx.symbol.Variable("data")
net <- mx.symbol.FullyConnected(data=net, name="fc1", num_hidden=128)
net2 <- mx.symbol.Variable("data2")
net2 <- mx.symbol.FullyConnected(data=net2, name="net2", num_hidden=128)
composed.net <- mx.apply(net, data=net2, name="compose")
arguments(composed.net)
```

```
## [1] "data2"       "net2_weight" "net2_bias"   "fc1_weight"  "fc1_bias"
```

In the example, *net* is used as a function to apply to an existing symbol
*net*. The resulting *composed.net* will replace the original argument *data* with
*net2* instead.

## Training a Neural Net

The [model API](https://github.com/apache/mxnet/blob/master/R-package/R/model.R) is a thin wrapper around the symbolic executors to support neural net training.

We encourage you to read [Symbolic Configuration and Execution in Pictures for python package](/api/python/symbol_in_pictures/symbol_in_pictures.md)for a detailed explanation of concepts in pictures.

## How Efficient Is the Symbolic API?

The Symbolic API brings the efficient C++
operations in powerful toolkits, such as CXXNet and Caffe, together with the
flexible dynamic NDArray operations. All of the memory and computation resources are
allocated statically during bind operations, to maximize runtime performance and memory
utilization.

The coarse-grained operators are equivalent to CXXNet layers, which are
extremely efficient.  We also provide fine-grained operators for more flexible
composition. Because MXNet does more in-place memory allocation, it can
be more memory efficient than CXXNet and gets to the same runtime with
greater flexibility.

## Next Steps
* [Classify Real-World Images with Pre-trained Model](/api/r/docs/tutorials/classify_real_image_with_pretrained_model)
* [Character Language Model using RNN](/api/r/docs/tutorials/char_rnn_model)
