---
layout: page_api
title: Scala Guide
action: Get Started
action_url: /get_started
permalink: /api/scala
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


# MXNet - Scala API

MXNet supports the Scala programming language. The MXNet Scala package brings flexible and efficient GPU
computing and state-of-art deep learning to Scala. It enables you to write seamless tensor/matrix computation with multiple GPUs in Scala. It also lets you construct and customize the state-of-art deep learning models in Scala, and apply them to tasks, such as image classification and data science challenges.



## Image Classification with the Scala Infer API
The Infer API can be used for single and batch image classification. More information can be found at the following locations:

## Tensor and Matrix Computations
You can perform tensor or matrix computation in pure Scala:

```scala
   import org.apache.mxnet._

   val arr = NDArray.ones(2, 3)
   // arr: org.apache.mxnet.NDArray = org.apache.mxnet.NDArray@f5e74790

   arr.shape
   // org.apache.mxnet.Shape = (2,3)

   (arr * 2).toArray
   // Array[Float] = Array(2.0, 2.0, 2.0, 2.0, 2.0, 2.0)

   (arr * 2).shape
   // org.apache.mxnet.Shape = (2,3)
```

## Related Resources

* [Neural Style in Scala on MXNet](https://github.com/apache/mxnet/blob/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/neuralstyle/NeuralStyle.scala)
* [More Scala Examples](https://github.com/apache/mxnet/tree/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples)
