mutable struct KVStore
  handle    :: MX_KVStoreHandle
  updater_c :: Ptr{Void}
  updater   :: Function

  KVStore(hdr :: MX_KVStoreHandle) = new(hdr, Ptr{Void}(0))
end

function KVStore(kv_type::Base.Symbol = :local)
  #@assert(kv_type ∈ [:local]) # TODO: update with allowed types

  ref_hdr = Ref{MX_handle}(0)
  kv_type = string(kv_type)
  @mxcall(:MXKVStoreCreate, (char_p, Ref{MX_handle}), kv_type, ref_hdr)
  return KVStore(MX_KVStoreHandle(ref_hdr[]))
end
function Base.unsafe_convert(::Type{MX_handle}, obj::KVStore)
  Base.unsafe_convert(MX_handle, obj.handle)
end
Base.convert(t::Type{MX_handle}, obj::KVStore) = Base.unsafe_convert(t, obj)
Base.cconvert(t::Type{MX_handle}, obj::KVStore) = Base.unsafe_convert(t, obj)

function _flatten_kvlist(keys :: Vector{Int}, vals :: Vector{Vector{NDArray}})
  @assert length(keys) == length(vals)
  keys_flt = Int[]
  vals_flt = NDArray[]
  for (k,v) in zip(keys, vals)
    append!(keys_flt, Base.ones(Int, length(v))*k)
    append!(vals_flt, v)
  end
  return (keys_flt, vals_flt)
end

function init!(self :: KVStore, key :: Int, val :: NDArray)
  init!(self, [key], [val])
end
function init!(self :: KVStore, key :: Int, vals :: Vector{NDArray})
  init!(self, Base.ones(Int, length(vals))*key, vals)
end
function init!(self :: KVStore, keys :: Vector{Int}, vals :: Vector{Vector{NDArray}})
  init!(self, _flatten_kvlist(keys, vals)...)
end
function init!(self :: KVStore, keys :: Vector{Int}, vals :: Vector{NDArray})
  @assert length(keys) == length(vals)
  keys = Cint[keys...]
  vals = MX_handle[vals...]
  @mxcall(:MXKVStoreInit, (MX_handle, MX_uint, Ptr{Cint}, Ptr{MX_handle}),
          self, length(keys), keys, vals)
end

import Base.push!
function push!(self :: KVStore, key :: Int, val :: NDArray; priority :: Int = 0)
  push!(self, [key], [val]; priority = priority)
end
function push!(self :: KVStore, key :: Int, vals :: Vector{NDArray}; priority :: Int = 0)
  push!(self, Base.ones(Int, length(vals))*key, vals; priority = priority)
end
function push!(self :: KVStore, keys :: Vector{Int}, vals :: Vector{Vector{NDArray}}; priority::Int=0)
  push!(self, _flatten_kvlist(keys, vals)...; priority = priority)
end
function push!(self :: KVStore, keys :: Vector{Int}, vals :: Vector{NDArray}; priority::Int=0)
  @assert length(keys) == length(vals)
  keys = Cint[keys...]
  vals = MX_handle[vals...]
  @mxcall(:MXKVStorePush, (MX_handle, MX_uint, Ptr{Cint}, Ptr{MX_handle}, Cint),
          self, length(keys), keys, vals, priority)
end

function pull!(self :: KVStore, key :: Int, out :: NDArray; priority :: Int = 0)
  pull!(self, [key], [out])
end
function pull!(self :: KVStore, key :: Int, outs :: Vector{NDArray}; priority :: Int = 0)
  pull!(self, Base.ones(Int, length(outs))*key, outs; priority = priority)
end
function pull!(self :: KVStore, keys :: Vector{Int}, outs :: Vector{Vector{NDArray}}; priority::Int=0)
  pull!(self, _flatten_kvlist(keys, outs)...; priority = priority)
end
function pull!(self :: KVStore, keys :: Vector{Int}, outs :: Vector{NDArray}; priority::Int=0)
  @assert length(keys) == length(outs)
  keys = Cint[keys...]
  outs = MX_handle[outs...]
  @mxcall(:MXKVStorePull, (MX_handle, MX_uint, Ptr{Cint}, Ptr{MX_handle}, Cint),
          self, length(keys), keys, outs, priority)
end


function get_type(self :: KVStore)
  type_ref = Ref{char_p}(0)
  @mxcall(:MXKVStoreGetType, (MX_handle, Ref{char_p}), self, type_ref)
  return Symbol(unsafe_string(type_ref[]))
end

function get_num_workers(self :: KVStore)
  ref_size = Ref{Cint}(0)
  @mxcall(:MXKVStoreGetGroupSize, (MX_handle, Ref{Cint}), self, ref_size)
  return Int(ref_size[])
end

function get_rank(self :: KVStore)
  ref_rank = Ref{Cint}(0)
  @mxcall(:MXKVStoreGetRank, (MX_handle, Ref{Cint}), self, ref_rank)
  return Int(ref_rank[])
end


# TODO: Currently Julia does not support closure in c-callbacks, so we are making use of the
# extra handle parameter of the API to pass the updater object around. Fix this when someday
# full closure cfunction is supported in Julia.
function _kvstore_update_wrapper(index::Cint, nd_recv::MX_handle, nd_local::MX_handle, updater::Ptr{Void})
  updater_func = unsafe_pointer_to_objref(updater) :: Function
  updater_func(Int(index), NDArray(MX_NDArrayHandle(nd_recv)), NDArray(MX_NDArrayHandle(nd_local)))
  return nothing
end
function set_updater(self :: KVStore, updater :: Function)
  self.updater = updater # keep a reference to the julia object so that updater_c is kept valid
  self.updater_c = cfunction(_kvstore_update_wrapper, Void, (Cint, MX_handle, MX_handle, Ptr{Void}))

  @mxcall(:MXKVStoreSetUpdater, (MX_handle, Ptr{Void}, Any),
          self, self.updater_c, updater)
end

function set_optimizer(self :: KVStore, optimizer :: AbstractOptimizer)
  ref_is_worker = Ref{Cint}(0)
  @mxcall(:MXKVStoreIsWorkerNode, (Ref{Cint},), ref_is_worker)
  is_worker = ref_is_worker[]

  if ismatch(r"dist", string(get_type(self))) && is_worker
    # TODO
  else
    set_updater(self, get_updater(optimizer))
  end
end
