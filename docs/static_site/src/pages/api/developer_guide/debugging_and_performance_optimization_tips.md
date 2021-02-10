---
layout: page_category
title:  Debugging and performance optimization tips
category: Developer Guide
permalink: /api/dev-guide/debugging_and_performance_optimization_tips
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

# Debugging and performance optimization tips

The general workflow when defining your network with Gluon API is either:

* build sequentially using `nn.Sequential` or `nn.HybridSequential` 

* inherit from `nn.Block` or `nn.HybridBlock`

## Debugging

When debugging your MXNet code, remember the following:

**Do NOT hybridize for debugging**

The difference between [imperative style (Gluon non-hybridized) and symbolic style (Gluon hybridized)]({{ "/versions/1.2.1/architecture/program_model.html" | relative_url }}) is:

* *imperative style* is _define-by-run_
* *symbolic style* is _define-then-run_


Basically, that means the execution path changes when calling `hybridize` on your network inherited from `HybridBlock` or `HybridSequential` (note that inheriting directly from `Block` is the same as not hybridizing your network). For efficiency, symbolic code does not keep the intermediate results and so it would be hard to debug and examine the intermediate outputs. Therefore, if you want to *examine the intermediate results for debugging, do NOT hybridize*. Once everything is working as expected, then you can `hybridize` and enjoy the speed up.

Please checkout the [d2l](http://d2l.ai/chapter_computational-performance/hybridize.html?highlight=hybridize#hybrid-programming) for more details about the hybrid-programming model.

## Use naive engine

It is also useful to set the environment variable `MXNET_ENGINE_TYPE='NaiveEngine'` prior to running your (end-to-end) code. This setting disables multi-threading and the execution engine will be synchronous, so you can examine the backtrace more easily. Remember to change it back to either the default `'ThreadedEnginePerDevice'` or `'ThreadedEngine'`.

For more details, here is a comprehensive tutorial on interactive debugging on [YouTube](https://www.youtube.com/watch?v=6-dOoJVw9_0).

## Performance optimization

Following up on using the environment variable `MXNET_ENGINE_TYPE` for debugging, here are the [available environment variables]({{ "/api/faq/env_var" | relative_url }})  that affect the performance of your code.

Please refer to [this presentation](https://www.slideshare.net/ThomasDelteil1/debugging-and-performance-tricks-for-mxnet-gluon) for more information on debugging and performance optimization.

