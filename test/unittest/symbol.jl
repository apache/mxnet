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

function test_compose()
  info("Symbol::compose")

  data = mx.variable(:data)
  net1 = mx.FullyConnected(data=data, name=:fc1, num_hidden=10)
  net1 = mx.FullyConnected(data=net1, name=:fc2, num_hidden=100)

  net2 = mx.FullyConnected(name=:fc3, num_hidden=10)
  net2 = mx.Activation(data=net2, act_type=:relu)
  net2 = mx.FullyConnected(data=net2, name=:fc4, num_hidden=20)

  composed  = net2(fc3_data=net1, name=:composed)
  multi_out = mx.group(composed, net1)
  @test mx.list_outputs(multi_out) == [:composed_output, :fc2_output]
end

################################################################################
# Run tests
################################################################################
test_basic()
test_internal()
test_compose()

end
