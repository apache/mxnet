module TestBind
using MXNet
if VERSION ≥ v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

using ..Main: rand_dims, reldiff

################################################################################
# Test Implementations
################################################################################
function test_arithmetic{T <: mx.DType}(::Type{T}, uf, gf)
  shape = rand_dims()
  info("Bind::arithmetic::$T::$uf::dims = $shape")

  lhs = mx.Variable(:lhs)
  rhs = mx.Variable(:rhs)
  ret = uf(lhs, rhs)
  @test mx.list_arguments(ret) == [:lhs, :rhs]

  lhs_arr  = mx.NDArray(rand(T, shape))
  rhs_arr  = mx.NDArray(rand(T, shape))
  lhs_grad = mx.empty(T, shape)
  rhs_grad = mx.empty(T, shape)

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
    test_arithmetic(T, .+, (g,x,y) -> (g,g))
    test_arithmetic(T, .-, (g,x,y) -> (g,-g))
    test_arithmetic(T, .*, (g,x,y) -> (y.*g, x.*g))
    if T <: Integer || T == Float16
      warn("Not running division test for $T")
    else
      test_arithmetic(T, ./, (g,x,y) -> (g ./ y, -x .* g ./ (y.^2)))
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

