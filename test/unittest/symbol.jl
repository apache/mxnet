module TestSymbol
using MXNet
using Base.Test

using ..Main: mlp2

################################################################################
# Test Implementations
################################################################################
function test_basic()
  info("Symbol::basic")

  model = mlp2()
  @test mx.list_arguments(model) == [:data,:fc1_weight,:fc1_bias,:fc2_weight,:fc2_bias]
  @test mx.list_outputs(model) == [:fc2_output]
  @test mx.list_auxiliary_states(model) == Symbol[]
end

function test_internal()
  info("Symbol::internal")

  data  = mx.variable(:data)
  oldfc = mx.FullyConnected(data=data, name=:fc1, num_hidden=10)
  net1  = mx.FullyConnected(data=oldfc, name=:fc2, num_hidden=100)

  @test mx.list_arguments(net1) == [:data,:fc1_weight,:fc1_bias,:fc2_weight,:fc2_bias]

  internal = mx.get_internals(net1)
  fc1      = internal[:fc1_output]
  @test mx.list_arguments(fc1) == mx.list_arguments(oldfc)
end

################################################################################
# Run tests
################################################################################
test_basic()
test_internal()

end
