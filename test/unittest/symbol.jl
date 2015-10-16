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

function test_infer_shape()
  info("Symbol::infer_shape::mlp2")

  model = mlp2()
  data_shape = (100, 100)
  arg_shapes, out_shapes, aux_shapes = mx.infer_shape(model, data=data_shape)
  arg_shape_dict = Dict{Symbol,Tuple}(zip(mx.list_arguments(model), arg_shapes))
  @test arg_shape_dict == Dict{Symbol,Tuple}(:fc2_bias => (10,),:fc2_weight => (1000,10),
                                             :fc1_bias => (1000,), :fc1_weight => (100, 1000),
                                             :data => data_shape)
  @test length(out_shapes) == 1
  @test out_shapes[1] == (10, 100)
end

function test_infer_shape_error()
  info("Symbol::infer_shape::throws")

  model = mlp2()
  weight_shape = (100, 1)
  data_shape   = (100, 100)
  @test_throws mx.MXError mx.infer_shape(model, data=data_shape, fc1_weight=weight_shape)
end


################################################################################
# Run tests
################################################################################
test_basic()
test_internal()
test_compose()
test_infer_shape()
test_infer_shape_error()

end
