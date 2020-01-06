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

module TestRandom
using MXNet
using Test
using Statistics

function test_uniform()
  dims = (100, 100, 2)
  @info "random::uniform::dims = $dims"

  low = -10; high = 10
  seed = 123
  mx.seed!(seed)
  ret1 = mx.rand(dims..., low = low, high = high)

  mx.seed!(seed)
  ret2 = NDArray(undef, dims)
  mx.rand!(ret2, low = low, high = high)

  @test copy(ret1) == copy(ret2)
  @test abs(mean(copy(ret1)) - (high+low)/2) < 0.1
end

function test_gaussian()
  dims = (80, 80, 4)
  @info "random::gaussian::dims = $dims"

  μ = 10; σ = 2
  seed = 456
  mx.seed!(seed)
  ret1 = mx.randn(dims..., μ = μ, σ = σ)

  mx.seed!(seed)
  ret2 = NDArray(undef, dims)
  mx.randn!(ret2, μ = μ, σ = σ)

  @test copy(ret1) == copy(ret2)
  @test abs(mean(copy(ret1)) - μ) < 0.1
  @test abs(std(copy(ret1)) - σ) < 0.1
end

@testset "Random Test" begin
  test_uniform()
  test_gaussian()
end

end
