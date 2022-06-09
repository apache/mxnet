---
layout: page_api
title: Python Guide
action: Get Started
action_url: /get_started
permalink: /api/python
tag: python
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

## Apache MXNet - Python API

Apache MXNet provides a comprehensive and flexible Python API to serve a broad community of developers with different levels of experience and wide ranging requirements. In this section, we provide an in-depth discussion of the functionality provided by various MXNet Python packages.


Apache MXNet’s Python API has two primary high-level packages*: the Gluon API and Module API. We recommend that new users start with the Gluon API as it’s more flexible and easier to debug. Underlying these high-level packages are the core packages of NDArray and Symbol.


NDArray works with arrays in an imperative fashion, i.e. you define how arrays will be transformed to get to an end result. Symbol works with arrays in a declarative fashion, i.e. you define the end result that is required (via a symbolic graph) and the MXNet engine will use various optimizations to determine the steps required to obtain this. With NDArray you have a great deal of flexibility when composing operations (as you can use Python control flow), and you can easily step through your code and inspect the values of arrays, which helps with debugging. Unfortunately, this comes at a performance cost when compared to Symbol, which can perform optimizations on the symbolic graph.


Module API is backed by Symbol, so, although it’s very performant, it’s also a little more restrictive. With the Gluon API, you can get the best of both worlds. You can develop and test your model imperatively using NDArray, a then switch to Symbol for faster model training and inference (if Symbol equivalents exist for your operations).


Code examples are placed throughout the API documentation and these can be run after importing MXNet as follows:

```python
>>> import mxnet as mx
```
