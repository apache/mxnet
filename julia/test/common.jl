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

################################################################################
# Common models used in testing
################################################################################
function rand_dims(max_ndim=6)
  tuple(rand(1:10, rand(1:max_ndim))...)
end

function mlp2()
  data = mx.Variable(:data)
  out = mx.FullyConnected(data, name=:fc1, num_hidden=1000)
  out = mx.Activation(out, act_type=:relu)
  out = mx.FullyConnected(out, name=:fc2, num_hidden=10)
  return out
end

function mlpchain()
  mx.@chain mx.Variable(:data) =>
            mx.FullyConnected(name=:fc1, num_hidden=1000) =>
            mx.Activation(act_type=:relu) =>
            mx.FullyConnected(name=:fc2, num_hidden=10)
end

"""
execution helper of SymbolicNode
"""
function exec(x::mx.SymbolicNode; feed...)
  ks, vs = zip(feed...)
  vs′ = mx.NDArray.(vs)

  e = mx.bind(x, context = mx.cpu(), args = Dict(zip(ks, vs′)))
  mx.forward(e)
  e.outputs
end
