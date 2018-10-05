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

# Autograd for NDArray
# this is a port of Python's autograd module
# https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/autograd.py

###############################################################################
#  Private util functions
###############################################################################

"""
    _set_recording(state::Bool)::Bool

Set status to recording/not recording. When recording, graph will be constructed
for gradient computation.

## Parameters

* `state::Bool`

## Returns

Previous state before this set
"""
function _set_recording(state::Bool)::Bool
  prev = Ref{Cint}(C_NULL)
  @mxcall(:MXAutogradSetIsRecording, (Cint, Ref{Cint}), state, prev)
  prev[]
end

_set_recording(::Void) = nothing

"""
Set status to training/predicting.
For example, Dropout will drop inputs randomly when
`train_mode = true` while simply passing through if `train_mode = false`.

## Parameters
* `train_mode::Bool`

## Returns

Previous state before this set.
"""
function _set_training(train_mode::Bool)::Bool
  prev = Ref{Cint}(C_NULL)
  @mxcall(:MXAutogradSetIsTraining, (Cint, Ref{Cint}), train_mode, prev)
  prev[]
end

_set_training(::Void) = nothing

###############################################################################
#  Public API
###############################################################################

"""
    is_recording()::Bool

Get status on recording/not recording.
"""
function is_recording()::Bool
  state = Ref{Cint}(C_NULL)
  @mxcall(:MXAutogradIsRecording, (Ref{Cint},), state)
  state[]
end

"""
    is_training()::Bool

Get status on recording/not recording.
"""
function is_training()::Bool
  state = Ref{Cint}(C_NULL)
  @mxcall(:MXAutogradIsTraining, (Ref{Cint},), state)
  state[]
end

@inline function _record(f, is_record::Union{Void,Bool}, train_mode::Union{Void,Bool})
  # Port from Python's `_RecordingStateScope` context manager
  # __enter__
  prev_is_record = _set_recording(is_record)
  prev_train_mode = _set_training(train_mode)

  try
    f()
  finally
    # __exit__
    if is_record != nothing && prev_is_record != is_record
      _set_recording(prev_is_record)
    end
    if train_mode != nothing && prev_train_mode != train_mode
      _set_recording(prev_train_mode)
    end
  end
end

"""
    record(f, train_mode = true)
    record(translates = true) do
      ...
    end

Returns an autograd recording scope context to be used in `do` block
and captures code that needs gradients to be calculated.

Parameter `train_mode::Bool` controls whether the forward pass is in training
or predicting mode.
This controls the behavior of some layers such as `Dropout`, `BatchNorm`.

!!! note
    When forwarding with `train_mode = false`, the corresponding backward
    should also use `train_mode = false`, otherwise gradient is undefined.

```julia
x = mx.NDArray([1 2; 3 4])
∇ = mx.attach_grad!(x)
y = mx.record() do
  2x
end
mx.backward!(y)

julia> ∇
2×2 mx.NDArray{Int64,2} @ CPU0:
 2  2
 2  2
```
"""
record(f, train_mode::Bool = true) = _record(f, true, train_mode)

"""
    pause(f, train_mode = false)
    pause(train_mode = false) do
      ...
    end

Create a scope context for codes that do not need gradients to be calculated.

```julia
record() do
  ...
  pause() do
    # testing, IO, gradient updates...
  end
end
```
"""
pause(f, train_mode::Bool = false) = _record(f, false, train_mode)

"""
    train_mode(f)
    train_mode() do
      ...
    end

Create a scope context in which forward pass behavior is set to training mode,
without changing the recording states.

```julia
y = model(x)
train_mode() do
  z = mx.Dropout(y)
  ...
end
```
"""
train_mode(f) = _record(f, nothing, true)

"""
    predict_mode(f)
    predict_mode() do
      ...
    end

Create a scope context in which forward pass behavior is set to inference mode,
without changing the recording states.

```julia
record() do
  y = model(x)
  predict_mode() do
    y = sampling(y)
  end
end
```
"""
predict_mode(f) = _record(f, nothing, false)

"""
    backward!(head,  head_grad;  retain_graph = false, train_mode = true)
    backward!(heads, head_grads; retain_graph = false, train_mode = true)

Compute the gradients of heads w.r.t previously marked variables.

## Parameters

- `head::NDArray`: output NDArray

- `head_grad::NDArray` or `Void`: gradient coefficient with respect to head.

- `heads::Vector{NDArray}`: a list of output NDArray

- `head_grads::Vector`: a list of gradient coefficient with respect ot heads.
  the element should be `NDArray` or `Void`

- `retain_graph::Bool`: whether to keep the graph after backward. e.g:
  If you want to differentiate the same graph twice,
  you need to pass `retain_graph=true`.

- `train_mode::Bool`: whether to do backward for training or predicting.
"""
backward!(head::NDArray, head_grad::NDArray; kws...) =
  backward!([head], [head_grad]; kws...)

backward!(head::NDArray, head_grad::Void = nothing; kws...) =
  backward!([head], head_grad; kws...)

function backward!(heads::VecOfNDArray, head_grad::Void;
                   retain_graph::Bool = false, train_mode::Bool = true)
  @mxcall(
    :MXAutogradBackwardEx,
    (MX_uint,
     Ptr{MX_handle},
     Ptr{MX_handle},
     MX_uint,
     Ptr{MX_handle},
     Cint,
     Cint,
     Cint,
     Ptr{MX_handle},
     Ptr{MX_handle}),
    length(heads),
    map(x -> x.handle, heads),
    C_NULL,
    0,
    C_NULL,
    retain_graph,
    false,  # create_graph
    train_mode,
    C_NULL,
    C_NULL)
end

function backward!(heads::VecOfNDArray, head_grads::Vector;
                   retain_graph::Bool = false, train_mode::Bool = true)
  output_handles = map(x -> x.handle, heads)
  ograd_handles  = map(head_grads) do x
    if x isa NDArray
      x.handle
    elseif x isa Void
      MX_handle(C_NULL)
    else
      throw(ArgumentError("element of head_grads should be NDArray or Void"))
    end
  end
  @assert length(output_handles) == length(ograd_handles)
  @mxcall(
    :MXAutogradBackwardEx,
    (MX_uint,
     Ptr{MX_handle},
     Ptr{MX_handle},
     MX_uint,
     Ptr{MX_handle},
     Cint,
     Cint,
     Cint,
     Ptr{MX_handle},
     Ptr{MX_handle}),
    length(output_handles),
    output_handles,
    ograd_handles,
    0,
    C_NULL,
    retain_graph,
    false,  # create_graph
    train_mode,
    C_NULL,
    C_NULL)
end

"""
    getgrad(arr::NDArray)

Returns the gradient buffer attached to this `NDArray`.
If the gradient buffer isn't attached yet, return `nothing`.
"""
function getgrad(arr::NDArray)
  out = Ref{MX_handle}(C_NULL)
  @mxcall(:MXNDArrayGetGrad, (MX_handle, Ref{MX_handle}), arr.handle, out)
  (out[] == C_NULL) ? nothing : NDArray(MX_NDArrayHandle(out[]))
end

"""
    attach_grad!(x::NDArray, grad_req::Symbol = :write)

Attach a gradient buffer to this `NDArray`,
so that [`backward!`](@ref) can compute gradient with respect to it.

## Parameters

- `x::NDArray`
- `grad_req::Symbol` (default is `:write`)

## Return

The attached gradient buffer

## See also

- [`getgrad`](@ref)
"""
function attach_grad!(x::NDArray, grad_req::Symbol = :write)
  # TODO: support storage type (stype in Python)
  # TODO: make sure it works with gpu array
  grad = zeros_like(x)
  _mark_variables!([x], [grad], grad_req)
  grad
end

"""
    mark_variables!(var,  grad,  grad_req)
    mark_variables!(vars, grads, grad_reqs)

Mark `NDArrays` as variables to compute gradient for autograd.

## Parameters

- `var::NDArray`
- `grad::NDArray`
- `grad_req::Symbol`: `:nop`, `:write`, `:inplace` or `:add`
- `vars::Vector{NDArray}`
- `grads::Vector{NDArray}`
- `grad_req::Vector{Symbol}`
"""
mark_variables!(var::NDArray, grad::NDArray, grad_reqs::Symbol = :write) =
  _mark_variables!([var], [grad], grad_reqs)

mark_variables!(var::VecOfNDArray, grads::VecOfNDArray, grad_reqs = :write) =
  _mark_variables!(var, grads, grad_reqs)

@inline function _getgrad_req(x::Symbol)::GRAD_REQ
  val = get(grad_req_map, x, false)
  if val == false
    throw(ArgumentError("invalid grad_reqs $x"))
  end
  val
end

@inline _getgrad_reqs(x::Symbol, n::Int) =
  map((_) -> MX_uint(_getgrad_req(x)), Base.OneTo(n))

@inline function _getgrad_reqs(xs::Vector{Symbol}, n::Int)
  if length(xs) != n
    throw(ArgumentError("number of variables and grad_reqs not matched"))
  end
  map(MX_uint ∘ _getgrad_req, xs)
end

@inline function _mark_variables!(vars::VecOfNDArray, grads::VecOfNDArray,
                                  grad_reqs = :write)
  n = length(vars)
  if n != length(grads)
    throw(ArgumentError("number of variables and gradients not matched"))
  end

  var_hdls  = map(x -> x.handle, vars)
  grad_hdls = map(x -> x.handle, grads)
  grad_reqs = _getgrad_reqs(grad_reqs, n)

  @mxcall(:MXAutogradMarkVariables,
          (MX_uint, Ref{MX_handle}, Ptr{MX_uint}, Ref{MX_handle}),
          length(vars), var_hdls, grad_reqs, grad_hdls)
end

"""
    symbol(x::NDArray)

Retrieve recorded computation history as `SymbolicNode`,
 where `x` is a `NDArray` representing the head of computation graph.
 """
function symbol(x::NDArray)
  ref = Ref{MX_handle}(C_NULL)
  @mxcall(:MXAutogradGetSymbol, (MX_handle, Ref{MX_handle}), x, ref)
  SymbolicNode(MX_SymbolHandle(ref[]))
end

###############################################################################
#  TODO: User-defined differentiable function
###############################################################################
