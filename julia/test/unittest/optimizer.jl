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

module TestOptimizer

using Test

using MXNet
using MXNet.mx.LearningRate
using MXNet.mx.Momentum


function test_fixed_η()
  @info "Optimizer::LearningRate::Fixed"
  x = LearningRate.Fixed(.42)
  @test get(x) == .42
  update!(x)
  @test get(x) == .42
end  # function test_fixed_η


function check_η_decay(x)
  @info "Optimizer::LearningRate::$x"

  η = get(x)
  @test η == 1

  for i ∈ 1:5
    update!(x)
    η′ = get(x)
    @test η′ < η
    η = η′
  end
end  # function check_η_decay


test_exp_η() = LearningRate.Exp(1) |> check_η_decay


test_inv_η() = LearningRate.Inv(1) |> check_η_decay


function test_μ_null()
  @info "Optimizer::Momentum::Null"
  x = Momentum.Null()
  @test iszero(get(x))
end


function test_μ_fixed()
  @info "Optimizer::Momentum::Fixed"
  x = Momentum.Fixed(42)
  @test get(x) == 42
end


@testset "Optimizer Test" begin
    @testset "LearningRate Test" begin
      test_fixed_η()
      test_exp_η()
      test_inv_η()
    end

    @testset "Momentum Test" begin
      test_μ_null()
      test_μ_fixed()
    end
end


end  # module TestOptimizer
