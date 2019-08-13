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
    grad(s::SymbolicNode, wrt::Vector{Symbol})

Get the autodiff gradient of the current `SymbolicNode`. This function can
only be used if the current symbol is a loss function.

# Arguments:
* `s::SymbolicNode`: current node.
* `wrt::Vector{Symbol}`: the names of the arguments to the gradient.

Returns a gradient symbol of the corresponding gradient.
"""
function grad(s::SymbolicNode, wrt::Vector{Symbol})
  hdr_ref = Ref{MX_handle}(C_NULL)
  keys = string.(key)

  @mxcall(:MXSymbolGrad, (MX_handle, MX_uint, char_pp, Ptr{MX_handle}),
          self, length(keys), keys, hdr_ref)
  return SymbolicNode(MX_SymbolHandle(hdr_ref[]))
end

function _build_shapes(shape_size::MX_uint, shape_ndim::Ptr{MX_uint}, shape_data::Ptr{Ptr{MX_uint}})
  shape_ndim = unsafe_wrap(Array, shape_ndim, shape_size)
  shape_data = unsafe_wrap(Array, shape_data, shape_size)
  shapes = map(1:shape_size) do i
    my_shape = unsafe_wrap(Array, shape_data[i], shape_ndim[i])
    tuple(reverse(Int[my_shape...], dims = 1)...)
  end
  convert(Vector{Tuple}, shapes)
end

function _infer_shape(self, keys, indptr, sdata)
  ref_arg_shape_size = Ref{MX_uint}(0)
  ref_arg_shape_ndim = Ref{Ptr{MX_uint}}(0)
  ref_arg_shape_data = Ref{Ptr{Ptr{MX_uint}}}(0)
  ref_out_shape_size = Ref{MX_uint}(0)
  ref_out_shape_ndim = Ref{Ptr{MX_uint}}(0)
  ref_out_shape_data = Ref{Ptr{Ptr{MX_uint}}}(0)
  ref_aux_shape_size = Ref{MX_uint}(0)
  ref_aux_shape_ndim = Ref{Ptr{MX_uint}}(0)
  ref_aux_shape_data = Ref{Ptr{Ptr{MX_uint}}}(0)
  ref_complete       = Ref{Cint}(0)
  @mxcall(:MXSymbolInferShape,
          (MX_handle, MX_uint, char_pp, Ptr{MX_uint}, Ptr{MX_uint},
           Ref{MX_uint}, Ref{Ptr{MX_uint}}, Ref{Ptr{Ptr{MX_uint}}},
           Ref{MX_uint}, Ref{Ptr{MX_uint}}, Ref{Ptr{Ptr{MX_uint}}},
           Ref{MX_uint}, Ref{Ptr{MX_uint}}, Ref{Ptr{Ptr{MX_uint}}},
           Ref{Cint}),
          self, length(indptr)-1, keys, indptr, sdata,
          ref_arg_shape_size, ref_arg_shape_ndim, ref_arg_shape_data,
          ref_out_shape_size, ref_out_shape_ndim, ref_out_shape_data,
          ref_aux_shape_size, ref_aux_shape_ndim, ref_aux_shape_data,
          ref_complete)
  if ref_complete[] == 0
    return (nothing, nothing, nothing)
  else
    return (
      _build_shapes(ref_arg_shape_size[], ref_arg_shape_ndim[], ref_arg_shape_data[]),
      _build_shapes(ref_out_shape_size[], ref_out_shape_ndim[], ref_out_shape_data[]),
      _build_shapes(ref_aux_shape_size[], ref_aux_shape_ndim[], ref_aux_shape_data[])
    )
  end
end

"""
    infer_shape(self :: SymbolicNode, args...)
    infer_shape(self :: SymbolicNode; kwargs...)

Do shape inference according to the input shapes. The input shapes could be provided
as a list of shapes, which should specify the shapes of inputs in the same order as
the arguments returned by [`list_arguments`](@ref). Alternatively, the shape information
could be specified via keyword arguments.

Returns a 3-tuple containing shapes of all the arguments, shapes of all the outputs and
shapes of all the auxiliary variables. If shape inference failed due to incomplete
or incompatible inputs, the return value will be `(nothing, nothing, nothing)`.
"""
function infer_shape(self :: SymbolicNode; kwargs...)
  sdata  = MX_uint[]
  indptr = MX_uint[0]
  for (k,v) in kwargs
    append!(sdata, reverse([v...], dims = 1))
    push!(indptr, length(sdata))
  end
  keys = AbstractString[string(x[1]) for x in kwargs]
  _infer_shape(self, keys, indptr, sdata)
end
function infer_shape(self :: SymbolicNode, args::Union{Tuple, Cvoid}...)
  sdata  = MX_uint[]
  indptr = MX_uint[0]
  for arg in args
    if isa(arg, Cvoid); continue; end
    append!(sdata, reverse([arg...], dims = 1))
    push!(indptr, length(sdata))
  end
  keys = Ptr{char_p}(0)
  _infer_shape(self, keys, indptr, sdata)
end

function _infer_type(self, keys, arg_type_data)
  ref_in_type_size  = Ref{MX_uint}()
  ref_in_type_data  = Ref{Ptr{Cint}}()
  ref_out_type_size = Ref{MX_uint}()
  ref_out_type_data = Ref{Ptr{Cint}}()
  ref_aux_type_size = Ref{MX_uint}()
  ref_aux_type_data = Ref{Ptr{Cint}}()
  ref_complete      = Ref{Cint}()

  @mxcall(:MXSymbolInferType,
          (MX_handle, MX_uint, char_pp, Ptr{Cint},
           Ref{MX_uint}, Ref{Ptr{Cint}},
           Ref{MX_uint}, Ref{Ptr{Cint}},
           Ref{MX_uint}, Ref{Ptr{Cint}},
           Ref{Cint}),
          self, length(arg_type_data)-1, keys, arg_type_data,
          ref_in_type_size, ref_in_type_data,
          ref_out_type_size, ref_out_type_data,
          ref_aux_type_size, ref_aux_type_data,
          ref_complete)

  if ref_complete[] == 0
    return (nothing, nothing, nothing)
  else
    in_type = unsafe_wrap(Array, ref_in_type_data[], ref_in_type_size[])
    out_type = unsafe_wrap(Array, ref_out_type_data[], ref_out_type_size[])
    aux_type = unsafe_wrap(Array, ref_aux_type_data[], ref_aux_type_size[])
    return ([fromTypeFlag(TypeFlag(t)) for t in in_type],
            [fromTypeFlag(TypeFlag(t)) for t in out_type],
            [fromTypeFlag(TypeFlag(t)) for t in aux_type])
  end
end

"""
    infer_type(self :: SymbolicNode; kwargs...)
    infer_type(self :: SymbolicNode, args...)

Do type inference according to the input types. The input types could be provided
as a list of types, which should specify the types of inputs in the same order as
the arguments returned by [`list_arguments`](@ref). Alternatively, the type information
could be specified via keyword arguments.

Returns a 3-tuple containing types of all the arguments, types of all the outputs and
types of all the auxiliary variables. If type inference failed due to incomplete
or incompatible inputs, the return value will be `(nothing, nothing, nothing)`.
"""
function infer_type(self :: SymbolicNode; kwargs...)
  types = Cint[toTypeFlag(x[2]) for x in kwargs]
  keys = AbstractString[string(x[1]) for x in kwargs]
  _infer_type(self, keys, types)
end

function infer_type(self :: SymbolicNode, args :: Union{Tuple,Cvoid}...)
  types = Cint[]
  keys = Ptr{char_p}(0)

  for arg in args
    if isa(arg, Cvoid); continue; end
    push!(types, toTypeFlag(arg))
  end
  _infer_type(self, keys, types)
end
