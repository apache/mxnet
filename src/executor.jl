type Executor
  handle :: MX_ExecutorHandle
  symbol :: Symbol
  arg_arrays  :: Vector{NDArray}
  grad_arrays :: Vector{Union{Void,NDArray}}
  aux_arrays  :: Vector{NDArray}
  out_arrays  :: Vector{NDArray}
end
function Executor(hdr :: MX_ExecutorHandle, symbol :: Symbol,
                  arg_arrays :: Vector{NDArray}, grad_arrays :: Vector{Union{Void,NDArray}},
                  aux_arrays :: Vector{NDArray})
  # get output arrays
  ref_size = Ref{MX_uint}
  ref_hdrs = Ref{Ptr{MX_handle}}
  @mxcall(:MXExecutorOutputs, (MX_handle, Ref{MX_uint}, Ref{Ptr{MX_handle}}),
          hdr, ref_size, ref_hdrs)
  out_hdrs = pointer_to_array(ref_hdrs[], ref_size[])
  out_arrays = [NDArray(MX_NDArrayHandle(x)) for x in out_hdrs]

  Executor(hdr, symbol, arg_arrays, grad_arrays, aux_arrays, out_arrays)
end

function Base.unsafe_convert(::Type{MX_handle}, obj::Executor)
  Base.unsafe_convert(MX_handle, obj.handle)
end
Base.convert(t::Type{MX_handle}, obj::Executor) = Base.unsafe_convert(t, obj)
Base.cconvert(t::Type{MX_handle}, obj::Executor) = Base.unsafe_convert(t, obj)

@enum GRAD_REQ GRAD_NULL=0 GRAD_WRITE=1 GRAD_ADD=3
function bind(self :: Symbol, ctx :: Context, args :: Union{Vector{NDArray},Dict{Base.Symbol,NDArray}};
              args_grad  :: Union{Void,Vector{NDArray},Dict{Base.Symbol,NDArray}} = nothing,
              aux_states :: Union{Void,Vector{NDArray},Dict{Base.Symbol,NDArray}} = nothing,
              grad_req   :: Union{GRAD_REQ,Vector{GRAD_REQ},Dict{Base.Symbol,GRAD_REQ}} = GRAD_WRITE)

  function get_ndarray_inputs(arg_key::String, args::Vector{NDArray}, arg_names::Vector{Base.Symbol}, allow_missing::Bool)
    @assert(length(args) == length(arg_names), "Length of $arg_key does not match number of arguments")
    return (MX_handle[args...], args)
  end
  function get_ndarray_inputs(arg_key::String, args::Dict{Base.Symbol,NDArray}, arg_names::Vector{Base.Symbol}, allow_missing::Bool)
    args_vec = map(arg_names) do name
      arr = get(args, name, nothing)
      if !allow_missing
        @assert(!isa(arr, Void), "Must specify all arguments in $arg_key ($name is missing)")
      end
      arr
    end
    args_hdr = MX_handle[(isa(x,Void) ? MX_handle(0) : x) for x in args_vec]
    return (args_hdr, args_vec)
  end

  arg_names = list_arguments(self)

  args_hdr, args = get_ndarray_inputs("args", args, arg_names, false)
  if isa(args_grad, Void)
    args_grad_hdr = MX_handle[Ptr{Void}(0) for i=1:length(args)]
  else
    args_grad_hdr, args_grad = get_ndarray_inputs("args_grad", args_grad, arg_names, true)
  end

  if isa(aux_states, Void); aux_states = NDArray[]; end
  aux_args_hdr, aux_states = get_ndarray_inputs("aux_states", aux_states, list_auxiliary_states(self), false)

  if isa(grad_req, GRAD_REQ)
    reqs = MX_uint[grad_req for i=1:length(args)]
  elseif isa(grad_req, Vector{GRAD_REQ})
    @assert(length(grad_req) == length(args))
    reqs = MX_uint[grad_req...]
  elseif isa(grad_req, Dict{Base.Symbol, GRAD_REQ})
    reqs = MX_uint[get(grad_req, name, GRAD_NULL) for name in arg_names]
  end

  ref_hdr = Ref{MX_handle}
  @mxcall(:MXExecutorBind,
          (MX_handle, Cint, Cint, MX_uint, Ptr{MX_handle}, Ptr{MX_handle}, Ptr{MX_uint},
           MX_uint, Ptr{MX_handle}, Ref{MX_handle}),
          self, ctx.device_type, ctx.device_id, length(args), args_hdr,
          args_grad_hdr, reqs, length(aux_states), uax_args_hdr, ref_hdr)
  executor = Executor(MX_ExecutorHandle(ref_hdr[]), self,
                      args, args_grad, aux_states)
end


function forward(self :: Executor; is_train::Bool=false, kwargs...)
  # TODO: kwargs
  @mxcall(:MXExecutorForward, (MX_handle, Cint), self, is_train)
end
