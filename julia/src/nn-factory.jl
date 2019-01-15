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

"""
    MLP(input, spec; hidden_activation = :relu, prefix)

Construct a multi-layer perceptron. A MLP is a multi-layer neural network with
fully connected layers.

# Arguments:
* `input::SymbolicNode`: the input to the mlp.
* `spec`: the mlp specification, a list of hidden dimensions. For example,
          `[128, (512, :sigmoid), 10]`. The number in the list indicate the
          number of hidden units in each layer. A tuple could be used to specify
          the activation of each layer. Otherwise, the default activation will
          be used (except for the last layer).
* `hidden_activation::Symbol`: keyword argument, default `:relu`, indicating
          the default activation for hidden layers. The specification here could be overwritten
          by layer-wise specification in the `spec` argument. Also activation is not
          applied to the last, i.e. the prediction layer. See [`Activation`](@ref) for a
          list of supported activation types.
* `prefix`: keyword argument, default `gensym()`, used as the prefix to
          name the constructed layers.

Returns the constructed MLP.
"""
function MLP(input, spec; hidden_activation::Symbol = :relu, prefix = gensym())
  spec = convert(Vector{Union{Int,Tuple}}, spec)

  n_layer = length(spec)
  for (i, s) in enumerate(spec)
    if isa(s, Tuple)
      n_unit, act_type = s
    else
      n_unit = s
      act_type = hidden_activation
    end
    input = FullyConnected(input, name=Symbol(prefix, "fc$i"), num_hidden=n_unit)
    if i < n_layer || isa(s, Tuple)
      # will not add activation unless the user explicitly specified
      input = Activation(input, name=Symbol(prefix, "$act_type$i"), act_type=act_type)
    end
  end

  return input
end
