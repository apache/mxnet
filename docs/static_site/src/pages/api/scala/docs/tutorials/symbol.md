---
layout: page_api
title: Symbol API
permalink: /api/scala/docs/tutorials/symbol
is_tutorial: true
tag: scala
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

# MXNet Scala Symbolic API

Topics:

* [How to Compose Symbols](#how-to-compose-symbols) introduces operator overloading of symbols.
* [Symbol Attributes](#symbol-attributes) describes how to attach attributes to symbols.
* [Serialization](#serialization) explains how to save and load symbols.
* [Executing Symbols](#executing-symbols) explains how to evaluate the symbols with data.
* [Execution API Reference]({{'/api/scala/docs/api/#org.apache.mxnet.Executor'|relative_url}}) documents the execution APIs.
* [Multiple Outputs](#multiple-outputs) explains how to configure multiple outputs.
* [Symbol Creation API Reference]({{'/api/scala/docs/api/#org.apache.mxnet.Symbol'|relative_url}}) documents functions.

We also highly encourage you to read [Symbolic Configuration and Execution in Pictures](symbol_in_pictures).

## How to Compose Symbols

The symbolic API provides a way to configure computation graphs.
You can configure the graphs either at the level of neural network layer operations or as fine-grained operations.

The following example configures a two-layer neural network.

```scala
    import org.apache.mxnet._
    val data = Symbol.Variable("data")
    val fc1 = Symbol.api.FullyConnected(Some(data), num_hidden = 128, name = "fc1")
    val act1 = Symbol.api.Activation(Some(fc1), "relu", "relu1")
    val fc2 = Symbol.api.FullyConnected(Some(act1), num_hidden = 64, name = "fc2")
    val net = Symbol.api.SoftmaxOutput(Some(fc2), name = "out")
    :type net
    // org.apache.mxnet.Symbol
```

The basic arithmetic operators (plus, minus, div, multiplication) are overloaded for
*element-wise operations* of symbols.

The following example creates a computation graph that adds two inputs together.

```scala
    import org.apache.mxnet._
    val a = Symbol.Variable("a")
    val b = Symbol.Variable("b")
    val c = a + b
```

## Symbol Attributes

You can add an attribute to a symbol by providing an attribute dictionary when you create a symbol.

```scala
    val data = Symbol.Variable("data", Map("mood"-> "angry"))
    val op = Symbol.api.Convolution(Some(data), kernel = Shape(1, 1), num_filter = 1, attr = Map("mood" -> "so so"))
```
For proper communication with the C++ backend, both the key and values of the attribute dictionary should be strings. To retrieve the attributes, use `attr(key)`:

```
    data.attr("mood")
    // Option[String] = Some(angry)
```

To attach attributes, you can use ```AttrScope```. ```AttrScope``` automatically adds the specified attributes to all of the symbols created within that scope. The user can also inherit this object to change naming behavior. For example:

```scala
    val (data, gdata) =
    AttrScope(Map("group" -> "4", "data" -> "great")).withScope {
      val data = Symbol.Variable("data", attr = Map("dtype" -> "data", "group" -> "1"))
      val gdata = Symbol.Variable("data2")
      (data, gdata)
    }
    assert(gdata.attr("group").get === "4")
    assert(data.attr("group").get === "1")

    val exceedScopeData = Symbol.Variable("data3")
    assert(exceedScopeData.attr("group") === None, "No group attr in global attr scope")
```

## Serialization

There are two ways to save and load the symbols. You can use the `mxnet.Symbol.save` and `mxnet.Symbol.load` functions to serialize the ```Symbol``` objects.
The advantage of using `save` and `load` functions is that it is language agnostic and cloud friendly.
The symbol is saved in JSON format. You can also get a JSON string directly using `mxnet.Symbol.toJson`.
Refer to [API documentation]({{'/api/scala/docs/api/#org.apache.mxnet.Symbol'|relative_url}}) for more details.

The following example shows how to save a symbol to an S3 bucket, load it back, and compare two symbols using a JSON string.

```scala
    import org.apache.mxnet._
    val a = Symbol.Variable("a")
    val b = Symbol.Variable("b")
    val c = a + b
    c.save("s3://my-bucket/symbol-c.json")
    val c2 = Symbol.load("s3://my-bucket/symbol-c.json")
    c.toJson == c2.toJson
    // Boolean = true
```

## Executing Symbols

After you have assembled a set of symbols into a computation graph, the MXNet engine can evaluate them.
If you are training a neural network, this is typically
handled by the high-level [Model class](model) and the [`fit()`] function.

For neural networks used in "feed-forward", "prediction", or "inference" mode (all terms for the same
thing: running a trained network), the input arguments are the
input data, and the weights of the neural network that were learned during training.

To manually execute a set of symbols, you need to create an [`Executor`] object,
which is typically constructed by calling the [`simpleBind(<parameters>)`] method on a symbol.

## Multiple Outputs

To group the symbols together, use the [mxnet.symbol.Group](#mxnet.symbol.Group) function.

```scala
    import org.apache.mxnet._
    val data = Symbol.Variable("data")
    val fc1 = Symbol.api.FullyConnected(Some(data), num_hidden = 128, name = "fc1")
    val act1 = Symbol.api.Activation(Some(fc1), "relu", "relu1")
    val fc2 = Symbol.api.FullyConnected(Some(act1), num_hidden = 64, name = "fc2")
    val net = Symbol.api.SoftmaxOutput(Some(fc2), name = "out")
    val group = Symbol.Group(fc1, net)
    group.listOutputs()
    // IndexedSeq[String] = ArrayBuffer(fc1_output, out_output)
```

After you get the ```group```, you can bind on ```group``` instead.
The resulting executor will have two outputs, one for fc1_output and one for softmax_output.

## Next Steps
* See [IO Data Loading API](io) for parsing and loading data.
* See [NDArray API](ndarray) for vector/matrix/tensor operations.
* See [KVStore API](kvstore) for multi-GPU and multi-host distributed training.
