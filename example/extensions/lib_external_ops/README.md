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

External Operators Example and Tutorial
=======================================

## Introduction

Extending MXNet with custom components used to mean distributing a custom fork. This feature allows adding custom components to MXNet by dynamically loading external libraries at runtime.

## Getting Started

### Have MXNet Ready

For this tutorial, clone MXNet from source and build it.

### Run An Example

This example shows compiling a custom backend operator and then dynamically loading it into MXNet at runtime. Go to the **lib_external_ops** directory and follow these steps:

1. Touch or modify the **min_ex.cc** and/or **min_ex-inl.h** file(s)
2. Go into the **build** directory that was created when building MXNet.
3. Run `make external_lib`. Notice that **libexternal_lib.so** has been rebuilt
4. Go to the **example/extensions/lib_external_ops** directory again and run `python test_loading.py`.