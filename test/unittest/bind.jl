module TestBind
using MXNet
using Base.Test

using ..Main: rand_dims, reldiff

################################################################################
# Test Implementations
################################################################################
function test_arithmetic(uf, gf)
  shape = rand_dims()
  info("Bind::arithmetic::$uf::dims = $shape")

  lhs = mx.variable(:lhs)
  rhs = mx.variable(:rhs)
  ret = uf(lhs, rhs)
  @test mx.list_arguments(ret) == [:lhs, :rhs]

  lhs_arr  = mx.NDArray(rand(shape))
  rhs_arr  = mx.NDArray(rand(shape))
  lhs_grad = mx.empty(shape)
  rhs_grad = mx.empty(shape)

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
  @test reldiff(out1, out2) < 1e-6
  @test reldiff(out1, out3) < 1e-6
  @test reldiff(out1, out4) < 1e-6

  # test gradients
  out_grad = mx.NDArray(ones(shape))
  lhs_grad2, rhs_grad2 = gf(copy(out_grad), copy(lhs_arr), copy(rhs_arr))
  mx.backward(exec2, out_grad)
  @test reldiff(copy(lhs_grad), lhs_grad2) < 1e-6
  @test reldiff(copy(rhs_grad), rhs_grad2) < 1e-6

  # reset grads
  lhs_grad[:] = 0
  rhs_grad[:] = 0
  # compute using another binding
  mx.backward(exec4, out_grad)
  @test reldiff(copy(lhs_grad), lhs_grad2) < 1e-6
  @test reldiff(copy(rhs_grad), rhs_grad2) < 1e-6
end

function test_arithmetic()
  test_arithmetic(.+, (g,x,y) -> (g,g))
  test_arithmetic(.-, (g,x,y) -> (g,-g))
  test_arithmetic(.*, (g,x,y) -> (y.*g, x.*g))
  test_arithmetic(./, (g,x,y) -> (g ./ y, -x .* g ./ (y.^2)))
end

################################################################################
# Run tests
################################################################################
test_arithmetic()

end

