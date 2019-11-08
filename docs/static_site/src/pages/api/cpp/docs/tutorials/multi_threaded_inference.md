--
layout: page_api
title: Multi Threaded Inference
action: Get Started
action_url: /get_started
permalink: /api/cpp/docs/tutorials/multi_threaded_inference
is_tutorial: true
tag: cpp
--
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

## Multi Threaded Inference API

A long standing request from MXNet users has been to invoke parallel inference on a model from multiple threads while sharing the parameters.
With this use case in mind, the threadsafe version of CachedOp was added to provide a way for customers to do multi-threaded inference for MXNet users.
This doc attempts to do the following:
1. Explain how one can use C API along with CPP package to achieve multithreaded inference. This will be useful for end users as well as frontend developers of different language bindings
2. Discuss the limitations of the above approach
3. Discuss the current state of thread safety in MXNet
4. Future TODOs

## Multithreaded inference in MXNet with C API and CPP Package

### Prerequisites
To complete this tutorial you need to:
- Learn the basics about [MXNet C++ API](/api/cpp)

## Setup the MXNet C++ API
To use the C++ API in MXNet, you need to build MXNet from source with C++ package. Please follow the [built from source guide](/get_started/ubuntu_setup.html), and [C++ Package documentation](/api/cpp)
The summary of those two documents is that you need to build MXNet from source with `USE_CPP_PACKAGE` flag set to 1. For example: `make -j USE_CPP_PACKAGE=1`.

## Download the model


## Current Limitations

## Current state of Thread Safety in MXNet

## Future TODOs
