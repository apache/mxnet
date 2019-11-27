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

using Base.Meta: isexpr
using Base.GC  # FIXME

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

_set_recording(::Cvoid) = nothing

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

_set_training(::Cvoid) = nothing

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

@inline function _record(f, is_record::Union{Cvoid,Bool}, train_mode::Union{Cvoid,Bool})
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

- `head_grad::NDArray` or `Nothing`: gradient coefficient with respect to head.

- `heads::Vector{NDArray}`: a list of output NDArray

- `head_grads::Vector`: a list of gradient coefficient with respect ot heads.
  the element should be `NDArray` or `Cvoid`

- `retain_graph::Bool`: whether to keep the graph after backward. e.g:
  If you want to differentiate the same graph twice,
  you need to pass `retain_graph=true`.

- `train_mode::Bool`: whether to do backward for training or predicting.
"""
backward!(head::NDArray, head_grad::NDArray; kws...) =
  backward!([head], [head_grad]; kws...)

backward!(head::NDArray, head_grad::Nothing = nothing; kws...) =
  backward!([head], head_grad; kws...)

function backward!(heads::VecOfNDArray, ::Nothing;
                   retain_graph::Bool = false, train_mode::Bool = true)
  cblist_ref = first(keys(_cblists))

  # TODO check MXAutogradBackwardEx usage
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
     Ptr{Ptr{MX_handle}},
     Ptr{Ptr{Cint}}),
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
    elseif x ≡ nothing  # faster than `x isa Cvoid` in Julia 0.7
      MX_handle(C_NULL)
    else
      throw(ArgumentError("element of head_grads should be NDArray or Cvoid"))
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
     Ptr{Ptr{MX_handle}},
     Ptr{Ptr{Cint}}),
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
#  User-defined differentiable function
###############################################################################


# gc-free holder
const _cbs_r  = [Ref{Ptr{Cvoid}}(C_NULL), Ref{Ptr{Cvoid}}(C_NULL)]
const _cbs    = [Ptr{Cvoid}(C_NULL), Ptr{Cvoid}(C_NULL)]
const _cbsref = Ref{Ptr{Ptr{Cvoid}}}(C_NULL)
const _frefs  = Dict()  # hold custom function instance and its args
const _conds  = []

function _back_wrapper(num_ograds, num_igrads, ptrs, reqs, is_train, fptr::Ptr{Cvoid})
  # @info "_back_wrapper"
  # hdls = unsafe_wrap(Array, ptrs, num_ograds + num_igrads)
  # @info "_back_wrapper" hdls
  # ograds = map(x -> NDArray(MX_NDArrayHandle(x), false), hdls[1:num_ograds])
  # @info "_back_wrapper" ograds
  # igrads = map(NDArray ∘ MX_NDArrayHandle, hdls[num_ograds+1:num_ograds+num_igrads])
  # @info "_back_wrapper" igrads
  # reqs = unsafe_wrap(Array, reqs, num_igrads)
  # @info "_back_wrapper" reqs
  #
  # # passing closure via raw pointer
  # f = unsafe_pointer_to_objref(fptr)
  #
  # Δs = backward!(f, ograds...)
  # Δs = Δs isa NDArray ? [Δs] : Δs
  #
  # # update gradient
  # for (i, Δ, req) ∈ zip(igrads, Δs, reqs)
  #   req = GRAD_REQ(req)
  #   if req == GRAD_NOP
  #     continue
  #   elseif req ∈ (GRAD_WRITE, GRAD_INPLACE)
  #     i[:] = Δ
  #   elseif req == GRAD_ADD
  #     i[:] += Δ
  #   end
  # end
  #
  # # release ref for gc
  # delete!(_frefs, f)

  Cint(true)
end

function _back_wrapper(num_ograds, num_igrads, ptrs, reqs, is_train, handle)
  ccall(:uv_async_send, Cint, (Ptr{Cvoid},), handle)
end

function _del_wrapper(handle)
  ccall(:uv_async_send, Cint, (Ptr{Cvoid},), handle)
end

function _wtf_wrapper(handle)
  ccall(:uv_async_send, Cint, (Ptr{Cvoid},), handle)
end

function _init_customfunc()  # will be invoked in __init__
  global _cbs_r
  global _cbs
  global _cbsref

  # the callback function prototype:
  # https://github.com/apache/incubator-mxnet/blob/ca565a00285d4fb0ca77ba9dc651a07ce1f01b24/include/mxnet/c_api.h#L209-L212
  _cbs_r[1][] = _cbs[1] = @cfunction(_back_wrapper, Cint,
                                     (Cint, Cint, Ptr{Ptr{Cvoid}}, Ptr{Cint},
                                      Cint, Ptr{Cvoid}))
  # _cbs_r[1][] = _cbs[1] = @cfunction(_wtf_wrapper, Cvoid, (Ptr{Cvoid},))

  _cbs_r[2][] = _cbs[2] = @cfunction(_del_wrapper, Cint, (Ptr{Cvoid},))
  _cbsref[] = Base.unsafe_convert(Ptr{Ptr{Cvoid}}, _cbs)
  @info "_init_customfunc" _cbs _cbsref[]
end

struct MXCallbackList
  n::Cint                # int num_callbacks;
  cbs::Ptr{Ptr{Cvoid}}   # int (**callbacks)(Cvoid);
  ctxs::Ptr{Ptr{Cvoid}}  # void **contexts;

  # we must provide two callback functions
  # the first is backward function `_back_wrapper`
  # the second is delete callback `_del_wrapper`
  # https://github.com/apache/incubator-mxnet/blob/2f8c1e83f94e84a25a48d2cd43136030fb3f2d1e/include/mxnet/c_api.h#L174-L182

  # `ctxs` is a array which is same size as `cbs`
  # its elements will be passed as `state` for callback functions,
  # usually the last argument.
  # In our case, we will push the pointer of custom func instance as
  # first element of `ctxs`; the pointer of MXCallbackList instance as
  # the second element.
  # The purpose of first pointer is to pass closure into `cfunction`.
  # The second pointer is to free the reference of MXCallbackList,
  # and let the function instance be GC-ed properly.

  function MXCallbackList(f)  # where all args are Refs
    fr = Ref(f)
    push!(_fholder, fr)
    @info "f ref" Base.unsafe_convert(Ptr{Cvoid}, fr)
    cond = Base.AsyncCondition() do cond
      @info "real back callback"
      A = ones(10000000)
      for i ∈ 1:10000
        B = A * A
      end
      @info "long run op end"
    end
    cond2 = Base.AsyncCondition() do cond
      @info "real del callback"
    end
    push!(_conds, cond)
    push!(_conds, cond2)
    @info "conds" cond.handle cond2.handle
    ctxs = [
      cond.handle,
      cond2.handle,
    ]
    ctxsptr = Base.unsafe_convert(Ptr{Ptr{Cvoid}}, ctxs)
    cblist = new(length(ctxs), _cbsref[], ctxsptr)
    # get the reference, and make a self-reference in ctxs[2]
    cblist_ref = Ref{MXCallbackList}(cblist)
    ctxs[2] = Base.unsafe_convert(Ptr{Cvoid}, cblist_ref)
    # insert ref into a holder to prevent from being GC-ed.
    # hold `xs` and `ys` which is passed into `MXCustomFunctionRecord`.
    _cblists[cblist_ref] = Ref(ctxs)
    cblist_ref
  end
end

# hold MXCallbackList to prevent from gc
const _cblists = Dict{Ref{MXCallbackList},Ref}()
const _fholder = []

"""
    @custom
Create callable custom function.
All the position-arguments should be `NDArray`.
The return value should be a instance of your custom type.
Please checkout `examples/autograd/customfunc.jl` for example.
"""
macro custom(ex::Expr)
  fdef = splitdef(ex)  # by MacroTools
  sig = ex.args[1]
  body = esc(Expr(:let, Expr(:block), ex.args[2]))  # create a new scope via `let`

  # only extract symbols, get rid of all annotations and default values
  args = map(x -> esc(splitarg(x)[1]), fdef[:args])
  # forward(f, xs...)
  forward_expr = Expr(:call, :forward, :f, args...)
  # insert keyword args
  if !isempty(fdef[:kwargs])
    # only extract symbols, get rid of all annotations and default values
    kwargs = map(fdef[:kwargs]) do x
      sym = splitarg(x)[1]
      Expr(:kw, sym, esc(sym))
    end
    append!(forward_expr.args, kwargs)
  end

  # xs, FIXME: a list of NDArray from positional argument
  xs_len = length(args)
  xs_expr = Expr(:vect, args...)

  body′ = quote
    f, ys = _record(false, nothing) do
      f = $body  # f is the object instance
      ys = $forward_expr
      f, ys
    end

    !is_recording() && return ys

    xs = $xs_expr
    ys′ = ys isa NDArray ? [ys] : ys

    # struct MXCallbackList
    cblist_ref = MXCallbackList(f)

    # gc-free
    xsr, ysr = Ref(xs), Ref(ys′)
    _frefs[f] = (xsr, ysr)
    # @info _frefs

    @mxcall(
      :MXCustomFunctionRecord,
      (Cint,            # num_inputs
       Ref{MX_handle},  # inputs

       Cint,            # num_outputs
       Ref{MX_handle},  # outputs

       Ref{MXCallbackList}),  # callbacks
      $xs_len,
      xs,

      length(ys′),
      ys′,

      cblist_ref)

    @info "inputs xs"  Base.unsafe_convert(Ref{MX_handle}, xs)
    @info "outputs ys" Base.unsafe_convert(Ref{MX_handle}, ys′)


    ys
  end

  GC.enable(false)  # FIXME

  Expr(:function, esc(sig), body′)
end

# custom function should overload these functions.
# the # of forward return values is the inputs of backward!.
function forward end
function backward! end
