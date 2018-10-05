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

import Base.push!

"""
    KVStore(kv_type = :local)

For single machine training, there are two commonly used types:

- `local`: Copies all gradients to CPU memory and updates weights there.

- `device`: Aggregates gradients and updates weights on GPU(s).
  With this setting, the `KVStore` also attempts to use GPU peer-to-peer
  communication, potentially accelerating the communication.

For distributed training, `KVStore` also supports a number of types:

- `dist_sync`: Behaves similarly to `local` but with one major difference.
  With `dist_sync`, batch-size now means the batch size used on each machine.
  So if there are `n` machines and we use batch size ``b``,
  then `dist_sync` behaves like `local` with batch size `n * b`.

- `dist_device_sync`: Identical to `dist_sync` with the difference similar
  to `device` vs `local`.

- `dist_async`: Performs asynchronous updates.
  The weights are updated whenever gradients are received from any machine.
  No two updates happen on the same weight at the same time.
  However, the order is not guaranteed.
"""
mutable struct KVStore
  handle    :: MX_KVStoreHandle
  updater_c :: Ptr{Void}
  updater   :: Function

  KVStore(hdr::MX_KVStoreHandle) = new(hdr, Ptr{Void}(0))
end

function KVStore(kv_type::Symbol = :local)
  @assert kv_type ∈ (:local, :device, :dist_sync, :dist_device_sync, :dist_async)
  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXKVStoreCreate, (char_p, Ref{MX_handle}), dump_mx_param(kv_type), ref_hdr)
  KVStore(MX_KVStoreHandle(ref_hdr[]))
end

Base.unsafe_convert(::Type{MX_handle}, obj::KVStore) =
  Base.unsafe_convert(MX_handle, obj.handle)
Base.convert(t::Type{MX_handle}, obj::KVStore) = Base.unsafe_convert(t, obj)
Base.cconvert(t::Type{MX_handle}, obj::KVStore) = Base.unsafe_convert(t, obj)

Base.show(io::IO, kv::KVStore) =
    print(io, "mx.KVStore @ $(get_type(kv))")

function _flatten_kvlist(keys::Vector{Int}, vals::Vector{<:Vector{<:NDArray}})
  @assert length(keys) == length(vals)
  keys_flt = Int[]
  vals_flt = NDArray[]
  for (k,v) in zip(keys, vals)
    append!(keys_flt, Base.ones(Int, length(v))*k)
    append!(vals_flt, v)
  end
  return (keys_flt, vals_flt)
end

"""
    init!(kv::KVStore, key::Int, val::NDArray)
    init!(kv::KVStore, keys, vals)

Initializes a single or a sequence of key-value pairs into the store.

For each key, one must `init!` it before calling `push!` or `pull!`.
When multiple workers invoke `init!` for the same key, only
the value supplied by worker with rank `0` is used. This function returns
after data has been initialized successfully.

```jldoctest
julia> kv = KVStore(:local)
mx.KVStore @ local

julia> init!(kv, 42, mx.rand(2, 3))
```
"""
init!(kv::KVStore, key::Int, val::NDArray) = init!(kv, [key], [val])
init!(kv::KVStore, key::Int, vals::Vector{<:NDArray}) =
  init!(kv, Base.ones(Int, length(vals)) * key, vals)
init!(kv::KVStore, keys::Vector{Int}, vals::Vector{<:Vector{<:NDArray}}) =
  init!(kv, _flatten_kvlist(keys, vals)...)

function init!(kv::KVStore, keys::Vector{Int}, vals::VecOfNDArray)
  @assert length(keys) == length(vals)
  keys = Cint[keys...]
  vals = MX_handle[vals...]
  @mxcall(:MXKVStoreInit, (MX_handle, MX_uint, Ptr{Cint}, Ptr{MX_handle}),
          kv, length(keys), keys, vals)
end

"""
    push!(kv::KVStore, key,  val;  priority = 0)
    push!(kv::KVStore, key,  vals; priority = 0)
    push!(kv::KVStore, keys, vals; priority = 0)

Pushes a single or a sequence of key-value pairs into the store.

This function returns immediately after adding an operator to the engine.
The actual operation is executed asynchronously. If there are consecutive
pushes to the same key, there is no guarantee on the serialization of pushes.
The execution of a push does not guarantee that all previous pushes are
finished. There is no synchronization between workers by default.
One can use ``barrier()`` to sync all workers.

`push!` and `pull!` single `NDArray`:
```jldoctest
julia> kv = KVStore(:local)
mx.KVStore @ local

julia> x = mx.empty(2, 3);

julia> init!(kv, 3, x)

julia> push!(kv, 3, mx.ones(2, 3) * 8)

julia> pull!(kv, 3, x)

julia> x
2×3 mx.NDArray{Float32,2} @ CPU0:
 8.0  8.0  8.0
 8.0  8.0  8.0
```

Aggregate values and `push!`:
```jldoctest
julia> vals = [mx.ones((2, 3), gpu(0)) * 3, mx.ones((2, 3), gpu(1)) * 4];

julia> push!(kv, 3, vals)

julia> pull!(kv, 3, x)

julia> x
2×3 mx.NDArray{Float32,2} @ CPU0:
 7.0  7.0  7.0
 7.0  7.0  7.0
```

`push!` a list of key to single device:

```jldoctest
julia> keys = [4, 5];

julia> init!(kv, keys, [empty(2, 3), empty(2, 3)])

julia> push!(kv, keys, [x, x])

julia> y, z = empty(2, 3), empty(2, 3);

julia> pull!(kv, keys, [y, z])
```
"""
push!(kv::KVStore, key::Int, val::NDArray; priority::Int = 0) =
  push!(kv, [key], [val]; priority = priority)
push!(kv::KVStore, key::Int, vals::Vector{<:NDArray}; priority::Int = 0) =
  push!(kv, Base.ones(Int, length(vals)) * key, vals; priority = priority)
push!(kv:: KVStore, keys::Vector{Int}, vals::Vector{<:Vector{<:NDArray}};
      priority::Int = 0) =
  push!(kv, _flatten_kvlist(keys, vals)...; priority = priority)

function push!(kv::KVStore, keys::Vector{Int}, vals::Vector{<:NDArray}; priority::Int = 0)
  @assert length(keys) == length(vals)
  keys = Cint[keys...]
  vals = MX_handle[vals...]
  @mxcall(:MXKVStorePush, (MX_handle, MX_uint, Ptr{Cint}, Ptr{MX_handle}, Cint),
          kv, length(keys), keys, vals, priority)
end

""" Pulls a single value or a sequence of values from the store.

This function returns immediately after adding an operator to the engine.
Subsequent attempts to read from the `out` variable will be blocked until the
pull operation completes.

`pull` is executed asynchronously after all previous `pull` calls and only
the last `push` call for the same input key(s) are finished.

The returned values are guaranteed to be the latest values in the store.

See [`pull!`](@ref) for more examples.
"""
pull!(kv::KVStore, key::Int, out::NDArray; priority::Int = 0) =
  pull!(kv, [key], [out], priority = priority)
pull!(kv::KVStore, key::Int, outs::Vector{<:NDArray}; priority::Int = 0) =
  pull!(kv, Base.ones(Int, length(outs))*key, outs; priority = priority)
pull!(kv::KVStore, keys::Vector{Int}, outs::Vector{<:Vector{<:NDArray}};
      priority::Int = 0) =
  pull!(kv, _flatten_kvlist(keys, outs)...; priority = priority)

function pull!(kv::KVStore, keys::Vector{Int}, outs::Vector{<:NDArray}; priority::Int = 0)
  @assert length(keys) == length(outs)
  keys = Cint[keys...]
  outs = MX_handle[outs...]
  @mxcall(:MXKVStorePull, (MX_handle, MX_uint, Ptr{Cint}, Ptr{MX_handle}, Cint),
          kv, length(keys), keys, outs, priority)
end


function get_type(kv::KVStore)
  type_ref = Ref{char_p}(0)
  @mxcall(:MXKVStoreGetType, (MX_handle, Ref{char_p}), kv, type_ref)
  return Symbol(unsafe_string(type_ref[]))
end

function get_num_workers(kv::KVStore)
  ref_size = Ref{Cint}(0)
  @mxcall(:MXKVStoreGetGroupSize, (MX_handle, Ref{Cint}), kv, ref_size)
  return Int(ref_size[])
end

function get_rank(kv::KVStore)
  ref_rank = Ref{Cint}(0)
  @mxcall(:MXKVStoreGetRank, (MX_handle, Ref{Cint}), kv, ref_rank)
  return Int(ref_rank[])
end

"""
    barrier(kv::KVStore)

Invokes global barrier among all worker nodes.

For example, assume there are `n` machines. We would like machine `0` to first
`init` the values and then have all the workers `pull` the initialized value.
Before pulling, we can place invoke `barrier(kv)` to guarantee that the
initialization is finished.
"""
barrier(kv::KVStore) = @mxcall(:MXKVStoreBarrier, (MX_handle,), kv)


# TODO: Currently Julia does not support closure in c-callbacks, so we are making use of the
# extra handle parameter of the API to pass the updater object around. Fix this when someday
# full closure cfunction is supported in Julia.
function _kvstore_update_wrapper(key::Cint, nd_recv::MX_handle, nd_local::MX_handle,
                                 updater::Ptr{Void})
  updater_func = unsafe_pointer_to_objref(updater)
  updater_func(Int(key), NDArray(MX_NDArrayHandle(nd_recv)),
               NDArray(MX_NDArrayHandle(nd_local)))
  nothing
end

"""
    setupdater!(kv, updater)

Sets a `push!` updater into the store.

This function only changes the local store.
When running on multiple machines one must use `set_optimizer`.

```jldoctest
julia> update(key, val, orig) = mx.@inplace orig += val .* .2
update (generic function with 1 method)

julia> kv = KVStore(:local)
mx.KVStore @ local

julia> mx.setupdater!(kv, update)

julia> init!(kv, 42, mx.ones(2, 3))

julia> push!(kv, 42, mx.ones(2, 3))

julia> x = empty(2, 3);

julia> pull!(kv, 42, x)

julia> x
2×3 mx.NDArray{Float32,2} @ CPU0:
 1.2  1.2  1.2
 1.2  1.2  1.2
```
"""
function setupdater!(kv::KVStore, updater)
  kv.updater = updater # keep a reference to the julia object so that updater_c is kept valid
  kv.updater_c = cfunction(_kvstore_update_wrapper, Void,
                           (Cint, MX_handle, MX_handle, Ptr{Void}))
  @mxcall(:MXKVStoreSetUpdater, (MX_handle, Ptr{Void}, Any),
          kv, kv.updater_c, updater)
end

"""
    setoptimizer!(kv::KVStore, opt)

Registers an optimizer with the kvstore.

When using a single machine, this function updates the local optimizer.
If using multiple machines and this operation is invoked from a worker node,
it will serialized the optimizer with pickle and send it to all servers.
The function returns after all servers have been updated.

```jldoctest
julia> kv = KVStore()
mx.KVStore @ local

julia> W = mx.zeros(2, 3)  # 2×3 weight matrix
2×3 mx.NDArray{Float32,2} @ CPU0:
 0.0  0.0  0.0
 0.0  0.0  0.0

julia> init!(kv, 42, W)

julia> setoptimizer!(kv, SGD(η = .2))  # SGD with .2 as learning rate

julia> ∇W = mx.ones(2, 3)  # assume it's the gradient
2×3 mx.NDArray{Float32,2} @ CPU0:
 1.0  1.0  1.0
 1.0  1.0  1.0

julia> push!(kv, 42, ∇W)

julia> pull!(kv, 42, W)  # fetch weight and write back to `W`

julia> W
2×3 mx.NDArray{Float32,2} @ CPU0:
 -0.2  -0.2  -0.2
 -0.2  -0.2  -0.2
```
"""
function setoptimizer!(kv::KVStore, opt::AbstractOptimizer)
  if ismatch(r"dist", string(get_type(kv))) && _isworker()
    # TODO
    error("not implemented")
  else
    setupdater!(kv, getupdater(opt))
  end
end

function _isworker()::Bool
  ref = Ref{Cint}(0)
  @mxcall(:MXKVStoreIsWorkerNode, (Ref{Cint},), ref)
  ref_is_worker[]
end

# TODO: sparse support?
