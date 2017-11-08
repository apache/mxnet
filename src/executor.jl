"""
    Executor

An executor is a realization of a symbolic architecture defined by a `SymbolicNode`.
The actual forward and backward computation specified by the network architecture can
be carried out with an executor.
"""
mutable struct Executor
  handle :: MX_ExecutorHandle
  symbol :: SymbolicNode
  arg_arrays  :: Vector{NDArray}
  grad_arrays :: Vector{Union{Void,NDArray}}
  aux_arrays  :: Vector{NDArray}
  outputs     :: Vector{NDArray}
  arg_dict    :: Dict{Base.Symbol, NDArray}
  aux_dict    :: Dict{Base.Symbol, NDArray}
end
function Executor(hdr :: MX_ExecutorHandle, symbol :: SymbolicNode,
                  arg_arrays :: Vector{NDArray}, grad_arrays :: Vector{Union{Void,NDArray}},
                  aux_arrays :: Vector{NDArray})
  # get output arrays
  ref_size = Ref{MX_uint}(0)
  ref_hdrs = Ref{Ptr{MX_handle}}(0)
  @mxcall(:MXExecutorOutputs, (MX_handle, Ref{MX_uint}, Ref{Ptr{MX_handle}}),
          hdr, ref_size, ref_hdrs)
  out_hdrs = unsafe_wrap(Array, ref_hdrs[], ref_size[])
  out_arrays = [NDArray(MX_NDArrayHandle(x)) for x in out_hdrs]

  arg_names = list_arguments(symbol)
  @assert(length(arg_names) == length(unique(arg_names)), "Duplicated names in arguments: $arg_names")
  arg_dict = Dict{Base.Symbol,NDArray}(zip(arg_names, arg_arrays))

  aux_names = list_auxiliary_states(symbol)
  @assert(length(aux_names) == length(unique(aux_names)), "Duplicated names in auxiliary states: $aux_names")
  aux_dict = Dict{Base.Symbol,NDArray}(zip(aux_names, aux_arrays))

  Executor(hdr, symbol, arg_arrays, grad_arrays, aux_arrays, out_arrays, arg_dict, aux_dict)
end

function Base.unsafe_convert(::Type{MX_handle}, obj::Executor)
  Base.unsafe_convert(MX_handle, obj.handle)
end
Base.convert(t::Type{MX_handle}, obj::Executor) = Base.unsafe_convert(t, obj)
Base.cconvert(t::Type{MX_handle}, obj::Executor) = Base.unsafe_convert(t, obj)

function _get_ndarray_inputs(arg_key::AbstractString, args::Vector{NDArray}, arg_names::Vector{Base.Symbol}, allow_missing::Bool)
  @assert(length(args) == length(arg_names), "Length of $arg_key does not match number of arguments")
  return (MX_handle[args...], args)
end
function _get_ndarray_inputs(arg_key::AbstractString, args::Dict{Base.Symbol,NDArray}, arg_names::Vector{Base.Symbol}, allow_missing::Bool)
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
          network parameters (weights, bias, filters, etc.), data and labels. See [`list_arguments`](@ref)
          and [`infer_shape`](@ref).
* `args_grad`:
* `aux_states`:
* `grad_req`:
"""
function bind(self :: SymbolicNode, ctx :: Context, args :: Union{Vector{NDArray},Dict{Base.Symbol,NDArray}};
              args_grad  :: Union{Vector{NDArray},Dict{Base.Symbol,NDArray}} = Dict{Base.Symbol,NDArray}(),
              aux_states :: Union{Vector{NDArray},Dict{Base.Symbol,NDArray}} = Dict{Base.Symbol,NDArray}(),
              grad_req   :: Union{GRAD_REQ,Vector{GRAD_REQ},Dict{Base.Symbol,GRAD_REQ}} = GRAD_WRITE)

  arg_names = list_arguments(self)

  args_hdr, args           = _get_ndarray_inputs("args", args, arg_names, false)
  args_grad_hdr, args_grad = _get_ndarray_inputs("args_grad", args_grad, arg_names, true)
  aux_args_hdr, aux_states = _get_ndarray_inputs("aux_states", aux_states, list_auxiliary_states(self), false)

  if isa(grad_req, GRAD_REQ)
    reqs = MX_uint[grad_req for i=1:length(args)]
  elseif isa(grad_req, Vector{GRAD_REQ})
    @assert(length(grad_req) == length(args))
    reqs = MX_uint[grad_req...]
  elseif isa(grad_req, Dict{Base.Symbol, GRAD_REQ})
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
function bind(self :: SymbolicNode; kwargs...)
  kwargs = Dict(kwargs)
  @assert(haskey(kwargs, :args), "Must specify args")
  args = pop!(kwargs, :args)
  if haskey(kwargs, :context)
    context = pop!(kwargs, :context)
  else
    context = cpu()
  end
  bind(self, context, args; kwargs...)
end

function simple_bind(self :: SymbolicNode, ctx :: Context;
                     grad_req :: Union{GRAD_REQ, Dict{Symbol, GRAD_REQ}}=GRAD_WRITE,
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


function forward(self :: Executor; is_train::Bool=false, kwargs...)
  for (k,v) in kwargs
    @assert(k ∈ self.arg_dict, "Unknown argument $k")
    @assert(isa(v, NDArray), "Keyword argument $k must be an NDArray")
    copy!(self.arg_dict[k], v)
  end

  @mxcall(:MXExecutorForward, (MX_handle, Cint), self, is_train)
end

function backward(self :: Executor)
  backward(self, NDArray[])
end
function backward(self :: Executor, out_grad :: NDArray)
  backward(self, [out_grad])
end
function backward(self :: Executor, out_grads :: Vector{NDArray})
  out_grads = MX_handle[out_grads...]
  @mxcall(:MXExecutorBackward, (MX_handle, MX_uint, Ptr{MX_handle}), self, length(out_grads), out_grads)
end


function copy_params_from(self::Executor, arg_params::Dict{Base.Symbol,NDArray},
                          aux_params::Union{Void,Dict{Base.Symbol,NDArray}}=nothing;
                          allow_extra_params::Bool=false)
  for (name, array) in arg_params
    if haskey(self.arg_dict, name)
      copy!(self.arg_dict[name], array)
    else
      @assert(allow_extra_params, "Extra params $name not in the arguments")
    end
  end

  if !isa(aux_params, Void)
    for (name, array) in aux_params
      if haskey(self.aux_dict, name)
        copy!(self.aux_dict[name], array)
      else
        @assert(allow_extra_params, "Extra auxiliary state $name not recognized")
      end
    end
  end
end


"""
Get a debug string about internal execution plan.

Can be used to get an estimated about the memory cost.
```julia
  net = ... # Symbol
  dProvider = ... # DataProvider
  exec = mx.simple_bind(net, mx.cpu(), data=size(dProvider.data_batch[1]))
  dbg_str = mx.debug_str(exec)
  println(split(ref, ['\\n'])[end-2])
```
"""
function debug_str(self :: Executor)
  s_ref = Ref{Cstring}()
  @mxcall(:MXExecutorPrint, (MX_handle, Ptr{Cstring}), self.handle, s_ref)
  unsafe_string(s_ref[])
end
