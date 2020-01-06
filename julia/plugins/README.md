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

# Plugins of MXNet.jl

This directory contains *plugins* of MXNet.jl. A plugin is typically a component that could be part of MXNet.jl, but excluded from the `mx` namespace. The plugins are included here primarily for two reasons:

* To minimize the dependency of MXNet.jl on other optional packages.
* To serve as examples on how to extend some components of MXNet.jl.

The most straightforward way to use a plugin is to `include` the code. For example

```julia
include(joinpath(Pkg.dir("MXNet"), "plugins", "io", "svmlight.jl"))

provider = SVMLightProvider("/path/to/dataset", 100)
```
