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

TBD

## Getting Started

### Have MXNet Ready

For this tutorial, clone MXNet from source but dont build it yet.

### Run An Example

This example shows compiling a custom backend operator and then dynamically loading it into MXNet at runtime. Go to the **lib_external_ops** directory and follow these steps:

1. Copy **min_ex.cc** and **min_ex-inl.h** into the src/operator directory.
2. Build MXNet.
3. Find the **min_ex.cc.o** file and copy it back to the **lib_external_ops** directory.
4. Delete the **min_ex.cc** and **min_ex-inl.h** from the src/operator directory.
5. Clean the build folder.
6. Rebuild MXNet.
7. Run `make` in the **lib_external_ops** directory to produce the libmin_ex.so with your custom operator inside.
8. Run `python test_loading.py`.