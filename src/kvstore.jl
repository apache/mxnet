type KVStore
  handle :: MX_KVStoreHandle
end

function KVStore(kv_type::Base.Symbol = :local)
  @assert(kv_type ∈ [:local]) # TODO: update with allowed types

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
