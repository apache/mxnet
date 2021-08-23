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

Makefile Configuration of MShadow
=====
MShadow is a template library, you only need to include mshadow to use it. So this folder is not used to build mshadow library file.

However, mshadow is a flexible library that allows you to compile with different configurations. For example,
you can compile mshadow without CUDA, and specify your own choice of BLAS.
There are different compile flags that you might need to set in your own configuration.
This folder provides a Makefile script to help you do that.

Usage
=====
* Set the configurations via variables in your Makefile, see example in [../guide/config.mk](../guide/config.mk)
* include [mshadow.mk](mshadow.mk) in your Makefile
* mshadow.mk will give you compiler variables that you can include when compiling
  - Add MSHADOW_CFLAGS to the compile flags
  - Add MSHADOW_LDFLAGS to the linker flags
  - Add MSHADOW_NVCCFLAGS to the nvcc compile flags
* For example Makefile, see [../guide/Makefile](../guide/Makefile)
