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

module TestSymbolicNode

using MXNet
using Test

using ..Main: mlp2, mlpchain, exec

################################################################################
# Test Implementations
################################################################################
function test_basic()
  @info("SymbolicNode::basic")

  model = mlp2()
  @test mx.list_arguments(model) == [:data,:fc1_weight,:fc1_bias,:fc2_weight,:fc2_bias]
  @test mx.list_outputs(model) == [:fc2_output]
  @test mx.list_auxiliary_states(model) == Symbol[]
end

function test_chain()
  @info("SymbolicNode::chain")

  model = mlpchain()
  @test mx.list_arguments(model) == [:data,:fc1_weight,:fc1_bias,:fc2_weight,:fc2_bias]
  @test mx.list_outputs(model) == [:fc2_output]
  @test mx.list_auxiliary_states(model) == Symbol[]

  let layerconfig = [20, 10, 6]
    model = @mx.chain mx.Variable(:data) =>
      mx.MLP(layerconfig, prefix=:magic_) =>
      mx.LinearRegressionOutput(mx.Variable(:label))

    @test mx.list_arguments(model) == [
      :data,
      :magic_fc1_weight, :magic_fc1_bias,
      :magic_fc2_weight, :magic_fc2_bias,
      :magic_fc3_weight, :magic_fc3_bias,
      :label]
  end
end

function test_internal()
  @info("SymbolicNode::internal")

  data  = mx.Variable(:data)
  oldfc = mx.FullyConnected(data, name=:fc1, num_hidden=10)
  net1  = mx.FullyConnected(oldfc, name=:fc2, num_hidden=100)

  @test mx.list_arguments(net1) == [:data,:fc1_weight,:fc1_bias,:fc2_weight,:fc2_bias]

  internal = mx.get_internals(net1)
  fc1      = internal[:fc1_output]
  @test mx.list_arguments(fc1) == mx.list_arguments(oldfc)
end

function test_get_children()
  @info("SymbolicNode::get_children")

  let x = mx.Variable(:x), y = mx.Variable(:y)
    z = x + y
    @test length(mx.list_outputs(z)) == 1
    @test length(mx.list_outputs(mx.get_children(z))) == 2
    @test mx.list_outputs(mx.get_children(z)) == [:x, :y]
  end

  @info("SymbolicNode::get_children::on leaf")
  let x = mx.Variable(:x)
    @test mx.get_children(x) == nothing
  end
end  # test_get_children


function test_compose()
  @info("SymbolicNode::compose")

  data = mx.Variable(:data)
  net1 = mx.FullyConnected(data, name=:fc1, num_hidden=10)
  net1 = mx.FullyConnected(net1, name=:fc2, num_hidden=100)

  net2 = mx.FullyConnected(mx.SymbolicNode, name=:fc3, num_hidden=10)
  net2 = mx.Activation(net2, act_type=:relu)
  net2 = mx.FullyConnected(net2, name=:fc4, num_hidden=20)

  composed  = net2(fc3_data=net1, name=:composed)
  multi_out = mx.Group(composed, net1)
  @test mx.list_outputs(multi_out) == [:composed_output, :fc2_output]
end

function test_infer_shape()
  @info("SymbolicNode::infer_shape::mlp2")

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
  @info("SymbolicNode::infer_shape::throws")

  model = mlp2()
  weight_shape = (100, 1)
  data_shape   = (100, 100)
  @test_throws mx.MXError mx.infer_shape(model, data=data_shape, fc1_weight=weight_shape)
end

function test_saveload()
  @info("SymbolicNode::saveload::mlp2")

  model = mlp2()
  fname = tempname()
  mx.save(fname, model)
  model2 = mx.load(fname, mx.SymbolicNode)
  @test mx.to_json(model) == mx.to_json(model2)

  rm(fname)
end

function test_attrs()
  @info("SymbolicNode::Attributes")

  data = mx.Variable(:data)

  @test mx.get_name(data) == :data
  result = mx.get_attr(data, :test)
  @test ismissing(result)
  mx.set_attr(data, :test, "1.0")
  result = mx.get_attr(data, :test)
  @test !ismissing(result)
  @test result == "1.0"

  data2 = mx.Variable(:data2, attrs = Dict(:test => "hallo!"))
  @test mx.get_attr(data2, :test) == "hallo!"

  conv = mx.Convolution(data2, kernel = (1,1), num_filter = 1)
  @test ismissing(mx.get_attr(conv, :b))
  @test mx.get_name(conv) isa Symbol

  @test_throws MethodError mx.Variable(:data3, attrs = Dict(:test => "1.0", :test2 => 1.0))
  @test_throws MethodError mx.Convolution(data2, kernel = (1,1), num_filter = 1, attrs = Dict(:test => "1.0", :test2 => 1.0))
end

function test_functions()
  @info("SymbolicNode::Functions")
  data = mx.Variable(:data)
  typeof(mx.sum(data)) == mx.SymbolicNode
end

function test_reshape()
  @info("SymbolicNode::reshape(sym, dim...)")

  A = mx.NDArray(collect(1:24))
  x = mx.Variable(:x)
  y = mx.reshape(x, 2, 3, 4)
  e = mx.bind(y, mx.cpu(), Dict(:x => A))
  mx.forward(e)
  out = e.outputs[1]

  @test size(out) == (2, 3, 4)
  @test copy(out) == reshape(1:24, 2, 3, 4)

  @info("SymbolicNode::reshape(sym, dim)")

  A = mx.NDArray(collect(1:24))
  x = mx.Variable(:x)
  y = mx.reshape(x, (2, 3, 4))
  e = mx.bind(y, mx.cpu(), Dict(:x => A))
  mx.forward(e)
  out = e.outputs[1]

  @test size(out) == (2, 3, 4)
  @test copy(out) == reshape(1:24, 2, 3, 4)

  @info("SymbolicNode::reshape::reverse")

  A = mx.zeros(10, 5, 4)
  x = mx.Variable(:x)
  y = mx.reshape(x, -1, 0, reverse = true)
  e = mx.bind(y, mx.cpu(), Dict(:x => A))
  mx.forward(e)
  out = e.outputs[1]

  @test size(out) == (50, 4)

  @info("SymbolicNode::reshape::0")

  A = mx.zeros(2, 3, 4)
  x = mx.Variable(:x)
  y = mx.reshape(x, 4, 0, 2)
  e = mx.bind(y, mx.cpu(), Dict(:x => A))
  mx.forward(e)
  out = e.outputs[1]

  @test size(out) == (4, 3, 2)

  @info("SymbolicNode::reshape::-1")

  A = mx.zeros(2, 3, 4)
  x = mx.Variable(:x)
  y = mx.reshape(x, 6, 1, -1)
  e = mx.bind(y, mx.cpu(), Dict(:x => A))
  mx.forward(e)
  out = e.outputs[1]

  @test size(out) == (6, 1, 4)

  @info("SymbolicNode::reshape::-2")

  A = mx.zeros(2, 3, 4, 2)
  x = mx.Variable(:x)
  y = mx.reshape(x, 3, 2, -2)
  e = mx.bind(y, mx.cpu(), Dict(:x => A))
  mx.forward(e)
  out = e.outputs[1]

  @test size(out) == (3, 2, 4, 2)

  @info("SymbolicNode::reshape::-3")

  A = mx.zeros(2, 3, 4, 5)
  x = mx.Variable(:x)
  y = mx.reshape(x, -3, -3)
  e = mx.bind(y, mx.cpu(), Dict(:x => A))
  mx.forward(e)
  out = e.outputs[1]

  @test size(out) == (6, 20)

  @info("SymbolicNode::reshape::-4")

  A = mx.zeros(2, 3, 4)
  x = mx.Variable(:x)
  y = mx.reshape(x, 0, 0, -4, 2, 2)
  e = mx.bind(y, mx.cpu(), Dict(:x => A))
  mx.forward(e)
  out = e.outputs[1]

  @test size(out) == (2, 3, 2, 2)
end

function test_dot()
  @info("SymbolicNode::dot")
  x = mx.Variable(:x)
  y = mx.Variable(:y)
  z = mx.dot(x, y)
  z_exec = mx.bind(z, context = mx.cpu(),
                   args = Dict(:x => mx.ones((100, 2)), :y => mx.ones((2, 200))))
  mx.forward(z_exec)

  ret = copy(z_exec.outputs[1])
  @test size(ret) == (100, 200)
  @test ret ≈ 2*ones(100, 200)
end

function test_print()
  @info("SymbolicNode::print")
  io = IOBuffer()
  print(io, mx.Variable(:x))
  @test !isempty(String(take!(io)))
end

function test_misc()
  @info("SymbolicNode::Miscellaneous")
  # Test for #189
  a = mx.Variable("a")
  b = mx.Variable("b")
  symb = mx.ElementWiseSum(a, b)
end

function test_add()
  @info("SymbolicNode::elementwise add")
  let x = mx.Variable(:x), A = Float32[1 2; 3 4]
    let y = exec(x .+ 42; :x => A)[]
      @test size(y) == size(A)
      @test copy(y) == A .+ 42
    end

    let y = exec(42 .+ x; :x => A)[]
      @test size(y) == size(A)
      @test copy(y) == 42 .+ A
    end

    let y = exec(-1 .+ x .+ 42; :x => A)[]
      @test size(y) == size(A)
      @test copy(y) == -1 .+ A .+ 42
    end
  end

  let A = Float32[1 2; 3 4], B = Float32[2 4; 6 8]
    x = mx.Variable(:x)
    y = mx.Variable(:y)

    let z = x .+ y
      z = exec(z; :x => A, :y => B)[]

      @test size(z) == size(A)
      @test copy(z) == A .+ B
    end

    let z = y .+ x
      z = exec(z; :x => A, :y => B)[]

      @test size(z) == size(A)
      @test copy(z) == B .+ A
    end
  end
end  # function test_add

function test_minus()
  @info("SymbolicNode::elementwise minus")
  let x = mx.Variable(:x), A = Float32[1 2; 3 4]
    let y = exec(x .- 42; :x => A)[]
      @test size(y) == size(A)
      @test copy(y) == A .- 42
    end

    let y = exec(42 .- x; :x => A)[]
      @test size(y) == size(A)
      @test copy(y) == 42 .- A
    end

    let y = exec(-1 .- x .- 42; :x => A)[]
      @test size(y) == size(A)
      @test copy(y) == -1 .- A .- 42
    end

    let y = exec(-x; :x => A)[]
      @test size(y) == size(A)
      @test copy(y) == -A
    end
  end

  let A = Float32[1 2; 3 4], B = Float32[2 4; 6 8]
    x = mx.Variable(:x)
    y = mx.Variable(:y)

    let z = x .- y
      z = exec(z; :x => A, :y => B)[]

      @test size(z) == size(A)
      @test copy(z) == A .- B
    end

    let z = y .- x
      z = exec(z; :x => A, :y => B)[]

      @test size(z) == size(A)
      @test copy(z) == B .- A
    end
  end
end  # function test_minus

function test_mul()
  @info("SymbolicNode::elementwise mul")
  let x = mx.Variable(:x), A = Float32[1 2; 3 4]
    let y = exec(x .* 42; :x => A)[]
      @test size(y) == size(A)
      @test copy(y) == A .* 42
    end

    let y = exec(42 .* x; :x => A)[]
      @test size(y) == size(A)
      @test copy(y) == 42 .* A
    end

    let y = exec(-1 .* x .* 42; :x => A)[]
      @test size(y) == size(A)
      @test copy(y) == -1 .* A .* 42
    end
  end

  let A = Float32[1 2; 3 4], B = Float32[2 4; 6 8]
    x = mx.Variable(:x)
    y = mx.Variable(:y)

    let z = x .* y
      z = exec(z; :x => A, :y => B)[]

      @test size(z) == size(A)
      @test copy(z) == A .* B
    end

    let z = y .* x
      z = exec(z; :x => A, :y => B)[]

      @test size(z) == size(A)
      @test copy(z) == B .* A
    end
  end
end  # function test_mul

function test_div()
  @info("SymbolicNode::elementwise div")
  let x = mx.Variable(:x), A = Float32[1 2; 3 4]
    let y = exec(x ./ 42; :x => A)[]
      @test size(y) == size(A)
      @test copy(y) ≈ A ./ 42
    end

    let y = exec(42 ./ x; :x => A)[]
      @test size(y) == size(A)
      @test copy(y) ≈ 42 ./ A
    end

    let y = exec(-1 ./ x ./ 42; :x => A)[]
      @test size(y) == size(A)
      @test copy(y) ≈ -1 ./ A ./ 42
    end
  end

  let A = Float32[1 2; 3 4], B = Float32[2 4; 6 8]
    x = mx.Variable(:x)
    y = mx.Variable(:y)

    let z = x ./ y
      z = exec(z; :x => A, :y => B)[]

      @test size(z) == size(A)
      @test copy(z) ≈ A ./ B
    end

    let z = y ./ x
      z = exec(z; :x => A, :y => B)[]

      @test size(z) == size(A)
      @test copy(z) ≈ B ./ A
    end
  end
end  # function test_div

function test_power()
  @info("SymbolicNode::elementwise power")
  let x = mx.Variable(:x), A = Float32[1 2; 3 4]
    let y = exec(x .^ 42; :x => A)[]
      @test size(y) == size(A)
      @test copy(y) ≈ A .^ 42
    end

    let y = exec(42 .^ x; :x => A)[]
      @test size(y) == size(A)
      @test copy(y) ≈ 42 .^ A
    end
  end

  let A = Float32[1 2; 3 4], B = Float32[2 4; 6 8]
    x = mx.Variable(:x)
    y = mx.Variable(:y)

    let z = x .^ y
      z = exec(z; :x => A, :y => B)[]

      @test size(z) == size(A)
      @test copy(z) ≈ A .^ B
    end

    let z = y .^ x
      z = exec(z; :x => A, :y => B)[]

      @test size(z) == size(A)
      @test copy(z) ≈ B .^ A
    end
  end

  @info("SymbolicNode::power::e .^ x::x .^ e")
  let x = mx.Variable(:x), A = [0 0 0; 0 0 0]
    y = exec(ℯ .^ x; :x => A)[]
    @test copy(y) ≈ fill(1, size(A))
  end

  let x = mx.Variable(:x), A = Float32[1 2; 3 4]
    let y = ℯ .^ x
      z = exec(y; :x => A)[]
      @test copy(z) ≈ ℯ .^ A
    end

    let y = x .^ ℯ
      z = exec(y; :x => A)[]
      @test copy(z) ≈ A .^ ℯ
    end
  end

  @info("SymbolicNode::power::π .^ x::x .^ π")
  let x = mx.Variable(:x), A = Float32[1 2; 3 4]
    let y = π .^ x
      z = exec(y; :x => A)[]
      @test copy(z) ≈ π .^ A
    end

    let y = x .^ π
      z = exec(y; :x => A)[]
      @test copy(z) ≈ A .^ π
    end
  end
end  # function test_power

function test_get_name()
  @info("SymbolicNode::get_name::with get_internals")
  name = mx.get_name(mx.get_internals(mlp2()))  # no error
  @test occursin("Ptr", name)
end  # function test_get_name

function test_var()
  @info("SymbolicNode::var")
  x = @mx.var x
  @test x isa mx.SymbolicNode

  x′ = @mx.var x
  @test x.handle != x′.handle

  x, y, z = @mx.var x y z
  @test x isa mx.SymbolicNode
  @test y isa mx.SymbolicNode
  @test z isa mx.SymbolicNode
end  # test_var


################################################################################
# Run tests
################################################################################
@testset "SymbolicNode Test" begin
  test_basic()
  test_chain()
  test_internal()
  test_compose()
  test_infer_shape()
  test_infer_shape_error()
  test_saveload()
  test_attrs()
  test_functions()
  test_reshape()
  test_dot()
  test_print()
  test_misc()
  test_add()
  test_minus()
  test_mul()
  test_div()
  test_power()
  test_get_name()
  test_var()
end

end
