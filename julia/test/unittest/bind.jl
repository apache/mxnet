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

module TestBind
using MXNet
using Test

using ..Main: rand_dims

################################################################################
# Test Implementations
################################################################################
function test_arithmetic(::Type{T}, uf, gf) where T <: mx.DType
  shape = rand_dims()
  @info "Bind::arithmetic::$T::$uf::dims = $shape"

  lhs = mx.Variable(:lhs)
  rhs = mx.Variable(:rhs)
  ret = uf(lhs, rhs)
  @test mx.list_arguments(ret) == [:lhs, :rhs]

  lhs_arr  = NDArray(rand(T, shape))
  rhs_arr  = NDArray(rand(T, shape))
  lhs_grad = NDArray{T}(undef, shape)
  rhs_grad = NDArray{T}(undef, shape)

  exec2 = mx.bind(ret, mx.Context(mx.CPU), [lhs_arr, rhs_arr], args_grad=[lhs_grad, rhs_grad])
  exec3 = mx.bind(ret, mx.Context(mx.CPU), [lhs_arr, rhs_arr])
  exec4 = mx.bind(ret, mx.Context(mx.CPU), Dict(:lhs=>lhs_arr, :rhs=>rhs_arr),
                  args_grad=Dict(:rhs=>rhs_grad, :lhs=>lhs_grad))

  mx.forward(exec2)
  mx.forward(exec3)
  mx.forward(exec4)

  out1 = uf(copy(lhs_arr), copy(rhs_arr))
  out2 = copy(exec2.outputs[1])
  out3 = copy(exec3.outputs[1])
  out4 = copy(exec4.outputs[1])
  @test isapprox(out1, out2)
  @test isapprox(out1, out3)
  @test isapprox(out1, out4)

  # test gradients
  out_grad = mx.NDArray(ones(T, shape))
  lhs_grad2, rhs_grad2 = gf(copy(out_grad), copy(lhs_arr), copy(rhs_arr))
  mx.backward(exec2, out_grad)
  @test isapprox(copy(lhs_grad), lhs_grad2)
  @test isapprox(copy(rhs_grad), rhs_grad2)

  # reset grads
  lhs_grad[:] = 0
  rhs_grad[:] = 0
  # compute using another binding
  mx.backward(exec4, out_grad)
  @test isapprox(copy(lhs_grad), lhs_grad2)
  @test isapprox(copy(rhs_grad), rhs_grad2)
end

function test_arithmetic()
  for T in [mx.fromTypeFlag(TF) for TF in instances(mx.TypeFlag)]
    test_arithmetic(T, (x,y) -> x .+ y, (g,x,y) -> (g,g))
    test_arithmetic(T, (x,y) -> x .- y, (g,x,y) -> (g,-g))
    test_arithmetic(T, (x,y) -> x .* y, (g,x,y) -> (y.*g, x.*g))
    if T <: Integer || T == Float16
      @warn "Not running division test for $T"
    else
      test_arithmetic(T, (x,y) -> x ./ y, (g,x,y) -> (g ./ y, -x .* g ./ (y.^2)))
    end
  end
end

################################################################################
# Run tests
################################################################################
@testset "Bind Test" begin
  test_arithmetic()
end

end

