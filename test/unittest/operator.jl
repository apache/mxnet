module TestOperator
using MXNet
if VERSION ≥ v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

using ..Main: rand_dims, reldiff

function test_scalar_op()
  data  = mx.Variable(:data)
  shape = rand_dims()
  info("Operator::scalar_op::dims = $shape")

  data_jl  = 5ones(Float32, shape)
  arr_data = mx.copy(data_jl, mx.cpu())
  arr_grad = mx.zeros(shape)

  test = 2 ./ (4 - ((1+data+1)*2/5) - 0.2)
  exec_test = mx.bind(test, mx.cpu(), [arr_data], args_grad=[arr_grad])
  mx.forward(exec_test)
  out = copy(exec_test.outputs[1])
  jl_out1 = (4 - ((1+data_jl+1)*2/5) - 0.2)
  jl_out = 2 ./ jl_out1
  @test reldiff(copy(out), jl_out) < 1e-6

  out_grad = 2mx.ones(shape)
  jl_grad  = 2copy(out_grad) / 5
  jl_grad  = 2jl_grad ./ (jl_out1 .^ 2)
  mx.backward(exec_test, out_grad)
  @test reldiff(copy(arr_grad), jl_grad) < 1e-6
end

################################################################################
# Run tests
################################################################################

@testset "Operator Test" begin
  test_scalar_op()
end

end
