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

using MXNet
using Base.Test

# run test in the whole directory, latest modified files
# are run first, this makes waiting time shorter when writing
# or modifying unit-tests
function test_dir(dir)
  jl_files = sort(filter(x -> ismatch(r".*\.jl$", x), readdir(dir)), by = fn -> stat(joinpath(dir,fn)).mtime)
  map(reverse(jl_files)) do file
    include("$dir/$file")
  end
end

info("libmxnet version => $(mx.LIB_VERSION)")

include(joinpath(dirname(@__FILE__), "common.jl"))
@testset "MXNet Test" begin
  test_dir(joinpath(dirname(@__FILE__), "unittest"))

  # run the basic MNIST mlp example
  if haskey(ENV, "CONTINUOUS_INTEGRATION")
    @testset "MNIST Test" begin
      include(joinpath(Pkg.dir("MXNet"), "examples", "mnist", "mlp-test.jl"))
    end
  end
end
