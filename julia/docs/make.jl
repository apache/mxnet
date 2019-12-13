# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

using Documenter
using DocumenterMarkdown
using MXNet

"""
Return all files of a submodule

julia> listpages("ndarray")
15-element Array{String,1}:
 "ndarray.jl"
 "ndarray/activation.jl"
 "ndarray/arithmetic.jl"
 "ndarray/array.jl"
 ...
 "ndarray/statistic.jl"
 "ndarray/trig.jl"
 "ndarray/type.jl"
"""
listpages(x) =
  ["$x.jl"; joinpath.(x, readdir(joinpath(@__DIR__, "..", "src", x)))]

const api_pages = [
  "api/context.md",
  "api/ndarray.md",
  "api/symbolic-node.md",
  "api/model.md",
  "api/initializers.md",
  "api/optimizers.md",
  "api/callbacks.md",
  "api/metric.md",
  "api/io.md",
  "api/nn-factory.md",
  "api/executor.md",
  "api/kvstore.md",
  "api/visualize.md",
]

makedocs(
  sitename = "MXNet.jl",
  modules  = MXNet,
  doctest  = false,
  format   = Markdown(),
)
