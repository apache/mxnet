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

Base.show(io::IO, sym::SymbolicNode) =
  print(io, "$(typeof(sym)) $(get_name(sym))")

"""
    print([io::IO], sym::SymbolicNode)

Print the content of symbol, used for debug.

```julia
julia> layer = @mx.chain mx.Variable(:data)           =>
         mx.FullyConnected(name=:fc1, num_hidden=128) =>
         mx.Activation(name=:relu1, act_type=:relu)
MXNet.mx.SymbolicNode(MXNet.mx.MX_SymbolHandle(Ptr{Nothing} @0x000055b29b9c3520))

julia> print(layer)
Symbol Outputs:
        output[0]=relu1(0)
Variable:data
Variable:fc1_weight
Variable:fc1_bias
--------------------
Op:FullyConnected, Name=fc1
Inputs:
        arg[0]=data(0) version=0
        arg[1]=fc1_weight(0) version=0
        arg[2]=fc1_bias(0) version=0
Attrs:
        num_hidden=128
--------------------
Op:Activation, Name=relu1
Inputs:
        arg[0]=fc1(0)
Attrs:
        act_type=relu
```
"""
function Base.print(io::IO, sym::SymbolicNode)
  out = Ref{mx.char_p}(C_NULL)
  @mx.mxcall(:MXSymbolPrint, (mx.MX_SymbolHandle, Ref{mx.char_p}), sym.handle, out)
  print(io, unsafe_string(out[]))
end

Base.print(sym::SymbolicNode) = print(stdout, sym)


