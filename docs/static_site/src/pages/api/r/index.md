---
layout: page_api
title: R Guide
action: Get Started
action_url: /get_started
permalink: /api/r
tag: r
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


# MXNet - R API

See the [MXNet R Reference Manual](https://s3.amazonaws.com/mxnet-prod/docs/R/mxnet-r-reference-manual.pdf).

MXNet supports the R programming language. The MXNet R package brings flexible and efficient GPU
computing and state-of-art deep learning to R. It enables you to write seamless tensor/matrix computation with multiple GPUs in R. It also lets you construct and customize the state-of-art deep learning models in R,
  and apply them to tasks, such as image classification and data science challenges.

You can perform tensor or matrix computation in R:

```r
   > require(mxnet)
   Loading required package: mxnet
   > a <- mx.nd.ones(c(2,3))
   > a
        [,1] [,2] [,3]
   [1,]    1    1    1
   [2,]    1    1    1
   > a + 1
        [,1] [,2] [,3]
   [1,]    2    2    2
   [2,]    2    2    2
```
## Resources

* [MXNet R Reference Manual](https://s3.amazonaws.com/mxnet-prod/docs/R/mxnet-r-reference-manual.pdf)
