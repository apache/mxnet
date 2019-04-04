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

import Base: bind

"""
    Executor

An executor is a realization of a symbolic architecture defined by a `SymbolicNode`.
The actual forward and backward computation specified by the network architecture can
be carried out with an executor.
"""
mutable struct Executor
  handle :: MX_ExecutorHandle
  symbol :: SymbolicNode
  arg_arrays  :: VecOfNDArray
  grad_arrays :: Vector{Union{Void,<:NDArray}}
  aux_arrays  :: VecOfNDArray
  outputs     :: VecOfNDArray
  arg_dict    :: Dict{Symbol}
  aux_dict    :: Dict{Symbol}
end

function Executor(hdl::MX_ExecutorHandle, sym::SymbolicNode,
                  arg_arrays::VecOfNDArray, grad_arrays::AbstractVector,
                  aux_arrays::VecOfNDArray)
  # get output arrays
  ref_size = Ref{MX_uint}(0)
  ref_hdls = Ref{Ptr{MX_handle}}(C_NULL)
  @mxcall(:MXExecutorOutputs, (MX_handle, Ref{MX_uint}, Ref{Ptr{MX_handle}}),
          hdl, ref_size, ref_hdls)
  out_hdrs = unsafe_wrap(Array, ref_hdls[], ref_size[])
  out_arrays = [NDArray(MX_NDArrayHandle(x)) for x in out_hdrs]

  arg_names = list_arguments(sym)
  @assert(length(arg_names) == length(unique(arg_names)), "Duplicated names in arguments: $arg_names")
  arg_dict = Dict(zip(arg_names, arg_arrays))

  aux_names = list_auxiliary_states(sym)
  @assert(length(aux_names) == length(unique(aux_names)), "Duplicated names in auxiliary states: $aux_names")
  aux_dict = Dict(zip(aux_names, aux_arrays))

  Executor(hdl, sym, arg_arrays, grad_arrays, aux_arrays, out_arrays, arg_dict, aux_dict)
end

Base.unsafe_convert(::Type{MX_handle}, obj::Executor) =
  Base.unsafe_convert(MX_handle, obj.handle)
Base.convert(t::Type{MX_handle}, obj::Executor) = Base.unsafe_convert(t, obj)
Base.cconvert(t::Type{MX_handle}, obj::Executor) = Base.unsafe_convert(t, obj)

function _get_ndarray_inputs(arg_key::AbstractString, args::VecOfNDArray,
                             arg_names::Vector{Symbol}, allow_missing::Bool)
  @assert(length(args) == length(arg_names), "Length of $arg_key does not match number of arguments")
  return (MX_handle[args...], args)
end

function _get_ndarray_inputs(arg_key::AbstractString, args::Dict{Symbol},
                             arg_names::Vector{Symbol}, allow_missing::Bool)
  args_vec = map(arg_names) do name
    arr = get(args, name, nothing)
    if !allow_missing
      @assert(!isa(arr, Void), "Must specify all arguments in $arg_key ($name is missing)")
    end
    arr
  end
  # help the type inference
  if allow_missing
    args_vec = Union{NDArray,Void}[args_vec...]
  else
    args_vec = NDArray[args_vec...]
  end
  args_hdr = MX_handle[(isa(x,Void) ? MX_handle(0) : x) for x in args_vec]
  return (args_hdr, args_vec)
end

"""
    bind(sym, ctx, args; args_grad=Dict(), aux_states=Dict(), grad_req=GRAD_WRITE)

Create an `Executor` by binding a `SymbolicNode` to concrete `NDArray`.

# Arguments
* `sym::SymbolicNode`: the network architecture describing the computation graph.
* `ctx::Context`: the context on which the computation should run.
* `args`: either a list of `NDArray` or a dictionary of name-array pairs. Concrete
          arrays for all the inputs in the network architecture. The inputs typically include
          network parameters (weights, bias, filters, etc.), data and labels.
          See [`list_arguments`](@ref) and [`infer_shape`](@ref).
* `args_grad`: a `Vector` of `NDArray` or a `Dict` contains `NDArray`
* `aux_states`: a `Vector` of `NDArray` or a `Dict` contains `NDArray`
* `grad_req`: single value, a `Vector` of `GRAD_REQ` or a `Dict{Symbol,GRAD_REQ}`
"""
function bind(self::SymbolicNode, ctx::Context, args;
              args_grad = Dict{Symbol,NDArray}(),
              aux_states = Dict{Symbol,NDArray}(),
              grad_req = GRAD_WRITE)

  arg_names = list_arguments(self)

  args_hdr, args           = _get_ndarray_inputs("args", args, arg_names, false)
  args_grad_hdr, args_grad = _get_ndarray_inputs("args_grad", args_grad, arg_names, true)
  aux_args_hdr, aux_states = _get_ndarray_inputs("aux_states", aux_states, list_auxiliary_states(self), false)

  if isa(grad_req, GRAD_REQ)
    reqs = MX_uint[grad_req for i=1:length(args)]
  elseif isa(grad_req, Vector{GRAD_REQ})
    @assert(length(grad_req) == length(args))
    reqs = MX_uint[grad_req...]
  elseif isa(grad_req, Dict{Symbol, GRAD_REQ})
    reqs = MX_uint[get(grad_req, name, GRAD_NOP) for name in arg_names]
  end

  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXExecutorBind,
          (MX_handle, Cint, Cint, MX_uint, Ptr{MX_handle}, Ptr{MX_handle}, Ptr{MX_uint},
           MX_uint, Ptr{MX_handle}, Ref{MX_handle}),
          self, ctx.device_type, ctx.device_id, length(args), args_hdr,
          args_grad_hdr, reqs, length(aux_states), aux_args_hdr, ref_hdr)
  args_grad = convert(Vector{Union{Void,NDArray}}, args_grad)
  executor = Executor(MX_ExecutorHandle(ref_hdr[]), self,
                      args, args_grad, aux_states)
end

function bind(x::SymbolicNode; context::Context = cpu(), kwargs...)
  kwargs = Dict(kwargs)
  @assert(haskey(kwargs, :args), "Must specify args")
  args = pop!(kwargs, :args)
  bind(x, context, args; kwargs...)
end

function simple_bind(self::SymbolicNode, ctx::Context;
                     grad_req::Union{GRAD_REQ,Dict{Symbol,GRAD_REQ}} = GRAD_WRITE,
                     kwargs...)
  arg_shapes, out_shapes, aux_shapes = infer_shape(self; kwargs...)
  @assert(!isa(arg_shapes, Void), "Information not enough to perform complete shape inference")

  arg_arrays = NDArray[zeros(shape, ctx) for shape in arg_shapes]
  arg_names  = list_arguments(self)

  grad_arrays = Dict{Symbol,NDArray}()

  if grad_req != GRAD_NOP
    shapes = zip(arg_names, arg_shapes)

    # if not in provided data, should be parameters
    provided_data_names = [x[1] for x in kwargs]
    shapes = filter(x -> !in(x[1], provided_data_names), shapes)

    # Remove all gradients for nop params
    # if isa(grad_req, Dict{Symbol, GRAD_REQ})
    #  shapes = filter(x -> grad_req[x[1]] != GRAD_NOP,shapes)
    # end

    for (name, shape) in shapes
      grad_arrays[name] = zeros(shape, ctx)
    end
  end

  aux_arrays = [zeros(shape, ctx) for shape in aux_shapes]
  return bind(self, ctx, arg_arrays, args_grad=grad_arrays, grad_req=grad_req, aux_states=aux_arrays)
end


function forward(self::Executor; is_train::Bool = false, kwargs...)
  for (k,v) in kwargs
    @assert(k ∈ self.arg_dict, "Unknown argument $k")
    @assert(isa(v, NDArray), "Keyword argument $k must be an NDArray")
    copy!(self.arg_dict[k], v)
  end

  @mxcall(:MXExecutorForward, (MX_handle, Cint), self, is_train)

  self.outputs
end

backward(x::Executor) = backward(x, NDArray[])
backward(x::Executor, out_grad::NDArray) = backward(x, [out_grad])
backward(x::Executor, out_grads::VecOfNDArray) =
  @mxcall(:MXExecutorBackward, (MX_handle, MX_uint, Ptr{MX_handle}),
          x, length(out_grads), MX_handle[out_grads...])

function copy_params_from(self::Executor, arg_params::Dict{Symbol},
                          aux_params::Dict{Symbol} = Dict{Symbol,Any}();
                          allow_extra_params::Bool = false)
  for (name, array) in arg_params
    if haskey(self.arg_dict, name)
      copy!(self.arg_dict[name], array)
    else
      @assert(allow_extra_params, "Extra params $name not in the arguments")
    end
  end

  for (name, array) in aux_params
    if haskey(self.aux_dict, name)
      copy!(self.aux_dict[name], array)
    else
      @assert(allow_extra_params, "Extra auxiliary state $name not recognized")
    end
  end
end


Base.show(io::IO, x::Executor) =
  print(io, "mx.", split(string(typeof(x)), '.')[end], " ", x.handle.value)

"""
    print([io::IO], x::Executor)

Get a debug string about internal execution plan.

Can be used to get an estimated about the memory cost.

```julia
julia> x = mx.Variable(:x)
MXNet.mx.SymbolicNode x

julia> exec = mx.bind(x + 1, mx.cpu(), Dict(:x => mx.ones(2,3)))
mx.Executor Ptr{Void} @0x000055c3dee9eb30

julia> print(exec)
Symbol Outputs:
        output[0]=_plus_scalar0(0)
Variable:x
--------------------
Op:_plus_scalar, Name=_plus_scalar0
Inputs:
        arg[0]=x(0) version=0
Attrs:
        scalar=1.00000000e+00
Total 0 MB allocated
Total 11 TempSpace resource requested
```
"""
Base.print(io::IO, x::Executor) = print(io, debug_str(x))
Base.print(x::Executor)         = print(STDOUT, x)

function debug_str(x::Executor)
  s_ref = Ref{Cstring}(C_NULL)
  @mxcall(:MXExecutorPrint, (MX_handle, Ptr{Cstring}), x.handle, s_ref)
  unsafe_string(s_ref[])
end
