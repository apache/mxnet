---
layout: page_api
title: NDArray API
is_tutorial: true
tag: perl
permalink: /api/perl/docs/tutorials/ndarray
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

# NDArray API

## Overview

A `AI::MXNet::NDArray` is a multidimensional container of items of the same type and
size. Various methods for data manipulation and computation are provided.

```perl
pdl> $x = mx->nd->array([[1, 2, 3], [4, 5, 6]])
pdl> print $x->aspdl->shape
[3, 2]
pdl> $y = $x + mx->nd->ones($x->shape)*3
pdl> print $y->aspdl
[
 [4 5 6]
 [7 8 9]
]
pdl> $z = $y->as_in_context(mx->gpu(0))
pdl> print $z,"\n"
<AI::MXNet::NDArray 2x3 @gpu(0)>
```

A detailed tutorial is available at
[https://mxnet.io/tutorials/basic/ndarray.html](https://mxnet.io/tutorials/basic/ndarray.html).

Note: AI::MXNet::NDarray is similar to numpy.ndarray in some aspects. But the difference is not negligible. For example

- AI::MXNet::NDArray->T does real data transpose to return new a copied array, instead
     of returning a view of the input array.
- AI::MXNet::NDArray->dot performs dot between the last axis of the first input array
     and the first axis of the second input, while numpy.dot uses the second
     last axis of the input array.

In additional, NDArray supports GPU computation and various neural
network layers.

AI::MXNet::NDarray also provides almost same routines as AI::MXNet::symbol. Most
routines between these two packages share the same C++ operator source
codes. But AI::MXNet::NDarray differs from AI::MXNet::Symbol in several aspects:

- AI::MXNet::NDArray adopts imperative programming, namely sentences are executed
     step-by-step so that the results can be obtained immediately.
