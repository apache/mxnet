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
    AbstractDataProvider

The root type for all data provider. A data provider should implement the following interfaces:

* [`get_batch_size`](@ref)
* [`provide_data`](@ref)
* [`provide_label`](@ref)

As well as the Julia iterator interface (see [the Julia manual](http://docs.julialang.org/en/stable/manual/interfaces/)).
Normally this involves defining:

* `Base.eltype(provider) -> AbstractDataBatch`
* `Base.start(provider) -> AbstractDataProviderState`
* `Base.done(provider, state) -> Bool`
* `Base.next(provider, state) -> (AbstractDataBatch, AbstractDataProvider)`
"""
abstract type AbstractDataProvider end

"""
    get_batch_size(provider) -> Int

# Arguments:
* `provider::AbstractDataProvider`: the data provider.

Returns the mini-batch size of the provided data. All the provided data should have the same mini-batch size (i.e. the last dimension).
"""
get_batch_size

"""
    provide_data(provider) -> Vector{Tuple{Base.Symbol, Tuple}}

# Arguments:
* `provider::AbstractDataProvider`: the data provider.

Returns a vector of (name, shape) pairs describing the names of the data it provides, and the corresponding shapes.

"""
provide_data

"""
    provide_label(provider) -> Vector{Tuple{Base.Symbol, Tuple}}

# Arguments:
* `provider::AbstractDataProvider`: the data provider.

Returns a vector of (name, shape) pairs describing the names of the labels it provides, and the corresponding shapes.
"""
provide_label

"""
    AbstractDataProviderState

   Base type for data provider states.
"""
abstract type AbstractDataProviderState end

"""
    AbstractDataBatch

   Base type for a data mini-batch. It should implement the following interfaces:

* [`count_samples`](@ref)
* [`get_data`](@ref)
* [`get_label`](@ref)

The following utility functions will be automatically defined:

* [`get`](@ref)
* [`load_data!`](@ref)
* [`load_label!`](@ref)
"""
abstract type AbstractDataBatch end

"""
    count_samples(provider, batch) -> Int

# Arguments:
* `batch::AbstractDataBatch`: the data batch object.

Returns the number of samples in this batch. This number should be greater than 0, but less than or equal to the batch size. This is used to indicate at the end of the data set, there might not be enough samples for a whole mini-batch.

"""
count_samples

"""
    get_data(provider, batch) -> Vector{NDArray}

# Arguments:
* `provider::AbstractDataProvider`: the data provider.
* `batch::AbstractDataBatch`: the data batch object.

Returns a vector of data in this batch, should be in the same order as declared in `provide_data() <AbstractDataProvider.provide_data>`.

The last dimension of each `NDArray` should always match the batch_size, even when `count_samples` returns a value less than the batch size. In this case,      the data provider is free to pad the remaining contents with any value.
"""
get_data

"""
    get_label(provider, batch) -> Vector{NDArray}

# Arguments:
* `provider::AbstractDataProvider`: the data provider.
* `batch::AbstractDataBatch`: the data batch object.

Returns a vector of labels in this batch. Similar to [`get_data`](@ref).
"""
get_label

"""
    DataBatch

A basic subclass of `AbstractDataBatch`, that implement the interface by
accessing member fields.
"""
mutable struct DataBatch{T,S,N,M} <: AbstractDataBatch
  data  :: Vector{NDArray{T,N}}
  label :: Vector{NDArray{S,M}}
  count :: Int
end

count_samples(batch::DataBatch) = batch.count

get_data(::Provider, batch::DataBatch) where {Provider<:AbstractDataProvider} =
  batch.data

get_label(::Provider, batch::DataBatch) where {Provider<:AbstractDataProvider} =
  batch.label

"""
    SlicedNDArray

A alias type of `Tuple{UnitRange{Int},NDArray}`.
"""
const SlicedNDArray = Tuple{UnitRange{Int},<:NDArray}

function _load_general!(provider :: AbstractDataProvider, batch :: AbstractDataBatch,
                        targets :: Vector{<:Vector{<:SlicedNDArray}}, loader::Function)
  data = loader(provider, batch)
  for (d_src, d_targets) in zip(data, targets)
    for (slice_idx, d_dst) in d_targets
      copy!(d_dst, slice(d_src, slice_idx))
    end
  end
end

"""
    load_data!(provider, batch, targets)

# Arguments:
* `provider::AbstractDataProvider`: the data provider.
* `batch::AbstractDataBatch`: the data batch object.
* `targets::Vector{Vector{SlicedNDArray}}`: the targets to load data into.

The targets is a list of the same length as number of data provided by this provider.
Each element in the list is a list of `SlicedNDArray`. This list described a
spliting scheme of this data batch into different slices, each slice is specified by
a slice-ndarray pair, where *slice* specify the range of samples in the mini-batch
that should be loaded into the corresponding *ndarray*.

This utility function is used in data parallelization, where a mini-batch is splited
and computed on several different devices.
"""
function load_data!(provider :: AbstractDataProvider, batch :: AbstractDataBatch,
                    targets :: Vector{<:Vector{<:SlicedNDArray}})
  _load_general!(provider, batch, targets, get_data)
end

"""
    load_label!(provider, batch, targets)

* `provider::AbstractDataProvider provider`: the data provider.
* `batch::AbstractDataBatch batch`: the data batch object.
* `targets::Vector{Vector{SlicedNDArray}}`: the targets to load label into.

The same as [`load_data!`](@ref), except that this is for loading labels.
"""
function load_label!(provider :: AbstractDataProvider, batch :: AbstractDataBatch,
                     targets :: Vector{<:Vector{<:SlicedNDArray}})
  _load_general!(provider, batch, targets, get_label)
end

function load_data!(provider :: AbstractDataProvider, batch :: AbstractDataBatch,
                    targets :: Vector{<:NDArray})
  for (src, dst) in zip(get_data(provider, batch), targets)
    copy!(dst, src)
  end
end
function load_label!(provider :: AbstractDataProvider, batch :: AbstractDataBatch,
                     targets :: Vector{<:NDArray})
  for (src, dst) in zip(get_label(provider, batch), targets)
    copy!(dst, src)
  end
end

import Base.get
"""
    get(provider, batch, name) -> NDArray

* `provider::AbstractDataProvider`: the data provider.
* `batch::AbstractDataBatch`: the data batch object.
* `name::Symbol`: the name of the data to get, should be one of the names
  provided in either `provide_data() <AbstractDataProvider.provide_data>`
  or `provide_label() <AbstractDataProvider.provide_label>`.

Returns the corresponding data array corresponding to that name.
"""
function get(provider::AbstractDataProvider, batch::AbstractDataBatch, name::Symbol)
  for (idx, (k, s)) in enumerate(provide_data(provider))
    if name == k
      return get_data(provider, batch)[idx]
    end
  end
  for (idx, (k, s)) in enumerate(provide_label(provider))
    if name == k
      return get_label(provider, batch)[idx]
    end
  end
  error("$name is not provided by this data provider")
end

"""
    eachbatch(provider::AbstractDataProvider)

Allows you to perform operations on data every epoch. This is especially useful
when you need to perform real-time augmentation of the data.

# Arguments:
* `provider`: an instance of the custom DataProvider type. You must return this
instance after modifying its fields.

"""
eachbatch(provider::AbstractDataProvider) = provider

"""
    ArrayDataProvider

A convenient tool to iterate `NDArray` or Julia `Array`.

    ArrayDataProvider(data[, label]; batch_size, shuffle, data_padding, label_padding)

Construct a data provider from `NDArray` or Julia Arrays.

# Arguments:
* `data`: the data, could be
  * a `NDArray`, or a Julia Array. This is equivalent to `:data => data`.
  * a name-data pair, like `:mydata => array`, where `:mydata` is the name of the data
  * and `array` is an `NDArray` or a Julia Array.
  * a list of name-data pairs.

* `label`: the same as the `data` parameter. When this argument is omitted, the constructed provider will provide no labels.
* `batch_size::Int`: the batch size, default is 0, which means treating the whole array as a single mini-batch.
* `shuffle::Bool`: turn on if the data should be shuffled at every epoch.
* `data_padding::Real`: when the mini-batch goes beyond the dataset boundary, there might
  be less samples to include than a mini-batch. This value specify a scalar to pad the
  contents of all the missing data points.
* `label_padding::Real`: the same as `data_padding`, except for the labels.

TODO: remove `data_padding` and `label_padding`, and implement rollover that copies
the last or first several training samples to feed the padding.
"""
mutable struct ArrayDataProvider{T,N} <: AbstractDataProvider
  data_arrays   :: Vector{Array{T,N}}
  data_names    :: Vector{Symbol}
  label_arrays
  label_names   :: Vector{Symbol}
  batch_size    :: Int
  sample_count  :: Int
  shuffle       :: Bool
  data_padding  :: MX_float
  label_padding :: MX_float

  data_batch
  label_batch
end

# Julia's type system is sometimes very frustrating. You cannot specify a function
# with argument Vector{Pair} to expect to be matched when calling with the parameter
# [:foo => zeros(2,3), :bar => zeros(3)] because the type inference gives very specific
# results, about the parametric type in the Pair{T1,T2} type, thus does not match the
# generic Pair type. In general, Int <: Number but Vector{Int} <: Vector{Number} is not
# true. So let us just use Any here...
function ArrayDataProvider(data; batch_size::Int = 0, shuffle::Bool = false,
                           data_padding::Real = 0, label_padding::Real = 0)
  ArrayDataProvider(data, [], batch_size = batch_size, shuffle = shuffle,
                    data_padding = data_padding, label_padding = label_padding)
end

function ArrayDataProvider(data, label; batch_size::Int = 0, shuffle::Bool = false,
                           data_padding::Real = 0, label_padding::Real = 0)
  asarr(arr :: Array{T}) where {T} = convert(Array{MX_float}, arr)
  asarr(arr :: NDArray) = copy(arr)

  if isa(data, Union{NDArray, Array}) && eltype(data) <: Real
    data_names  = [:data]
    data_arrays = Array{MX_float}[asarr(data)]
  elseif isa(data, Pair)
    @assert isa(data.first, Base.Symbol) && isa(data.second, Union{NDArray, Array})
    data_names  = [data.first]
    data_arrays = Array{MX_float}[asarr(data.second)]
  elseif isa(data, Vector) || isa(data, Tuple)
    map(data) do d
      @assert isa(d, Pair) && isa(d.first, Base.Symbol) && isa(d.second, Union{NDArray, Array})
    end
    data_names  = Base.Symbol[d.first for d in data]
    data_arrays = Array{MX_float}[asarr(d.second) for d in data]
  else
    error("Invalid data argument type")
  end

  if isa(label, Union{NDArray, Array}) && eltype(label) <: Real
    label_names  = [:softmax_label]
    label_arrays = Array{MX_float}[asarr(label)]
  elseif isa(label, Pair)
    @assert isa(label.first, Base.Symbol) && isa(label.second, Union{NDArray, Array})
    label_names  = [label.first]
    label_arrays = Array{MX_float}[asarr(label.second)]
  elseif isa(label, Vector) || isa(label, Tuple)
    map(label) do d
      @assert isa(d, Pair) && isa(d.first, Base.Symbol) && isa(d.second, Union{NDArray, Array})
    end
    label_names  = Base.Symbol[d.first for d in label]
    label_arrays = Array{MX_float}[asarr(d.second) for d in label]
  else
    error("Invalid label argument type")
  end

  @assert length(data_arrays) > 0
  sample_count = size(data_arrays[1])[end]
  for i = 1:length(data_names)
    @assert(size(data_arrays[i])[end] == sample_count,
            "Number of samples in  $(data_names[i]) is mismatch with $(data_names[1])")
  end
  for i = 1:length(label_names)
    @assert(size(label_arrays[i])[end] == sample_count,
            "Number of samples in  $(label_names[i]) is mismatch with $(data_names[1])")
  end

  if batch_size == 0
    batch_size = sample_count
  end
  @assert 0 < batch_size <= sample_count

  function gen_batch_nds(arrs :: Vector{Array{MX_float}}, bsize :: Int)
    map(arrs) do arr
      shape = size(arr)
      empty(shape[1:end-1]..., bsize)
    end
  end

  data_batch  = gen_batch_nds(data_arrays, batch_size)
  label_batch = gen_batch_nds(label_arrays, batch_size)

  # reshape data and labels into 2D tensors, so that it is easier to work with them
  data_arrays = map(data_arrays) do arr
    reshape(arr, prod(size(arr)[1:end-1]), size(arr)[end])
  end
  label_arrays = map(label_arrays) do arr
    reshape(arr, prod(size(arr)[1:end-1]), size(arr)[end])
  end

  ArrayDataProvider(data_arrays, data_names, label_arrays, label_names, batch_size,
                    sample_count, shuffle, MX_float(data_padding), MX_float(label_padding),
                    data_batch, label_batch)
end

provide_data(provider::ArrayDataProvider) =
  collect(zip(provider.data_names, map(size, provider.data_batch)))

provide_label(provider::ArrayDataProvider) =
  collect(zip(provider.label_names, map(size, provider.label_batch)))

get_batch_size(provider::ArrayDataProvider) = provider.batch_size

struct ArrayDataProviderState <: AbstractDataProviderState
  curr_idx :: Int
end

Base.eltype(provider :: ArrayDataProvider) = ArrayDataProviderState

function Base.start(provider :: ArrayDataProvider)
  if provider.shuffle
    # re-shuffle all data
    idx_perm = randperm(provider.sample_count)
    provider.data_arrays = map(x->x[:,idx_perm], provider.data_arrays)
    provider.label_arrays = map(x->x[:,idx_perm], provider.label_arrays)
  end

  return ArrayDataProviderState(1)
end

Base.done(provider::ArrayDataProvider, state::ArrayDataProviderState) =
  state.curr_idx > provider.sample_count

struct ArrayDataBatch <: AbstractDataBatch
  idx :: UnitRange{Int}
end
function Base.next(provider :: ArrayDataProvider, state :: ArrayDataProviderState)
  idx = state.curr_idx:Base.min(state.curr_idx+provider.batch_size-1, provider.sample_count)
  return (ArrayDataBatch(idx), ArrayDataProviderState(idx.stop+1))
end

function count_samples(provider :: ArrayDataProvider, batch :: ArrayDataBatch)
  return length(batch.idx)
end

function get_data(provider :: ArrayDataProvider, batch :: ArrayDataBatch)
  for (src, dst) in zip(provider.data_arrays, provider.data_batch)
    copy_ignore_shape!(dst[1:length(batch.idx)], src[:, batch.idx])
    if length(batch.idx) < provider.batch_size
      dst[length(batch.idx)+1:provider.batch_size] = provider.data_padding
    end
  end
  return provider.data_batch
end
function get_label(provider :: ArrayDataProvider, batch :: ArrayDataBatch)
  for (src, dst) in zip(provider.label_arrays, provider.label_batch)
    copy_ignore_shape!(dst[1:length(batch.idx)], src[:, batch.idx])
    if length(batch.idx) < provider.batch_size
      dst[length(batch.idx)+1:provider.batch_size] = provider.label_padding
    end
  end
  return provider.label_batch
end


"""
    MXDataProvider

A data provider that wrap built-in data iterators from libmxnet. See below for
a list of built-in data iterators.
"""
mutable struct MXDataProvider <: AbstractDataProvider
  handle     :: MX_DataIterHandle
  data_shape :: Vector{Tuple{Symbol,Tuple}}
  label_shape:: Vector{Tuple{Symbol,Tuple}}
  batch_size :: Int

  # those two a auxiliary variables to help avoid calling reset
  # but still pre-fetch first batch to get shape information
  first_epoch:: Bool
  first_batch:: Bool
end

function _reset_data_iter(handle :: MX_DataIterHandle)
  @mxcall(:MXDataIterBeforeFirst, (MX_handle,), handle)
end
function _iter_next(handle :: MX_DataIterHandle)
  ref_ret = Ref{Cint}(0)
  @mxcall(:MXDataIterNext, (MX_handle, Ref{Cint}), handle, ref_ret)
  return Bool(ref_ret[])
end
function _get_data(handle :: MX_DataIterHandle)
  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXDataIterGetData, (MX_handle, Ref{MX_handle}), handle, ref_hdr)
  return NDArray(MX_NDArrayHandle(ref_hdr[]), false)
end
function _get_label(handle :: MX_DataIterHandle)
  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXDataIterGetLabel, (MX_handle, Ref{MX_handle}), handle, ref_hdr)
  return NDArray(MX_NDArrayHandle(ref_hdr[]), false)
end

function MXDataProvider(handle     :: MX_DataIterHandle;
                        data_name  :: Symbol = :data,
                        label_name :: Union{Symbol,Void} = :softmax_label,
                        kwargs...) # for convenience, we ignore the rest keyword arguments
  # init iterator, load the first batch and get shapes
  @assert(_iter_next(handle), "Failed to load the first batch in MXDataProvider")
  data_shape = Tuple{Base.Symbol, Tuple}[(data_name, size(_get_data(handle)))]
  if !isa(label_name, Void)
    label_shape = Tuple{Base.Symbol, Tuple}[(label_name::Base.Symbol, size(_get_label(handle)))]
  else
    label_shape = Tuple{Base.Symbol, Tuple}[]
  end

  MXDataProvider(handle, data_shape, label_shape, data_shape[1][2][end], true, true)
end

provide_data(provider::MXDataProvider) = provider.data_shape
provide_label(provider::MXDataProvider) = provider.label_shape
get_batch_size(provider::MXDataProvider) = provider.batch_size

mutable struct MXDataProviderState <: AbstractDataProviderState
  has_next :: Bool
end
struct MXDataBatch <: AbstractDataBatch
end

function Base.eltype(provider :: MXDataProvider)
  MXDataBatch
end
function Base.start(provider :: MXDataProvider)
  if !provider.first_epoch
    _reset_data_iter(provider.handle)
  else
    provider.first_epoch = false
  end

  return MXDataProviderState(true)
end
function Base.done(provider :: MXDataProvider, state :: MXDataProviderState)
  if provider.first_batch
    state.has_next = true
    provider.first_batch = false
  else
    state.has_next = _iter_next(provider.handle)
  end
  return !state.has_next
end
function Base.next(provider :: MXDataProvider, state :: MXDataProviderState)
  return (MXDataBatch(), state)
end

function get_data(provider :: MXDataProvider, batch :: MXDataBatch)
  return NDArray[_get_data(provider.handle)]
end
function get_label(provider :: MXDataProvider, batch :: MXDataBatch)
  return NDArray[_get_label(provider.handle)]
end
function count_samples(provider :: MXDataProvider, batch :: MXDataBatch)
  ref_pad = Ref{Cint}(0)
  @mxcall(:MXDataIterGetPadNum, (MX_handle, Ref{Cint}), provider.handle, ref_pad)
  return provider.batch_size - Int(ref_pad[])
end

function _get_iter_creators()
  n_ref = Ref{MX_uint}(0)
  h_ref = Ref{Ptr{MX_handle}}(0)
  @mxcall(:MXListDataIters, (Ref{MX_uint}, Ref{Ptr{MX_handle}}), n_ref, h_ref)

  return unsafe_wrap(Array, h_ref[], n_ref[])
end

function _get_iter_name(hdr :: MX_handle)
  ref_name      = Ref{char_p}(0)
  ref_desc      = Ref{char_p}(0)
  ref_narg      = Ref{MX_uint}(0)
  ref_arg_names = Ref{char_pp}(0)
  ref_arg_types = Ref{char_pp}(0)
  ref_arg_descs = Ref{char_pp}(0)

  @mxcall(:MXDataIterGetIterInfo,
          (MX_handle, Ref{char_p}, Ref{char_p}, Ref{MX_uint}, Ref{char_pp}, Ref{char_pp}, Ref{char_pp}),
          hdr, ref_name, ref_desc, ref_narg, ref_arg_names, ref_arg_types, ref_arg_descs)

  return Symbol(unsafe_string(ref_name[]))
end

const _iter_creator_cache = Dict{Symbol,MX_handle}()
function _populate_iter_creator_cache!()
  empty!(_iter_creator_cache)
  h_creators = _get_iter_creators()
  for handle in h_creators
    name = _get_iter_name(handle)
    _iter_creator_cache[name] = handle
  end
end

_get_iter_creator(name :: Symbol) = _iter_creator_cache[name]

function _define_data_iter_creator(hdr :: MX_handle)
  ref_name      = Ref{char_p}(0)
  ref_desc      = Ref{char_p}(0)
  ref_narg      = Ref{MX_uint}(0)
  ref_arg_names = Ref{char_pp}(0)
  ref_arg_types = Ref{char_pp}(0)
  ref_arg_descs = Ref{char_pp}(0)

  @mxcall(:MXDataIterGetIterInfo,
          (MX_handle, Ref{char_p}, Ref{char_p}, Ref{MX_uint}, Ref{char_pp}, Ref{char_pp}, Ref{char_pp}),
          hdr, ref_name, ref_desc, ref_narg, ref_arg_names, ref_arg_types, ref_arg_descs)

  iter_name = Symbol(unsafe_string(ref_name[]))

  isprovider =  endswith(string(iter_name), "Iter")
  signature = _format_signature(Int(ref_narg[]), ref_arg_names)
  f_desc = "    " * string(iter_name) * "(" *signature * ")\n\n"
  if isprovider
    f_desc *= "Can also be called with the alias `$(string(iter_name)[1:end-4] * "Provider")`.\n"
  end
  f_desc *= unsafe_string(ref_desc[]) * "\n\n"
  f_desc *= "# Arguments:\n"
  f_desc *= "* `data_name::Symbol`: keyword argument, default `:data`. The name of the data.\n"
  f_desc *= "* `label_name::Symbol`: keyword argument, default `:softmax_label`. " *
            "The name of the label. Could be `nothing` if no label is presented in this dataset.\n\n"
  f_desc *= _format_docstring(Int(ref_narg[]), ref_arg_names, ref_arg_types, ref_arg_descs) * "\n"
  f_desc *= "Returns the constructed `MXDataProvider`."

  if isprovider
    alias_name = Symbol(string(iter_name)[1:end-4] * "Provider")
  else
    alias_name = nothing
  end

  defun = quote
    @doc $f_desc ->
    function $iter_name(; kwargs...)
      arg_keys = String[string(k) for (k,v) in kwargs]
      arg_vals = String[dump_mx_param(v) for (k,v) in kwargs]
      ref_hdr  = Ref{MX_handle}(0)

      local hdr = _get_iter_creator($(QuoteNode(iter_name)))
      @mxcall(:MXDataIterCreateIter, (MX_handle, MX_uint, char_pp, char_pp, Ref{MX_handle}),
              hdr, length(arg_keys), arg_keys, arg_vals, ref_hdr)

      return MXDataProvider(MX_DataIterHandle(ref_hdr[]); kwargs...)
    end
    $(isprovider ? :(const $alias_name = $iter_name) : :())

  end
  defun
end

macro _import_io_iterators()
  creators = _get_iter_creators()
  defs = Expr[]
  for handle in creators
    push!(defs, _define_data_iter_creator(handle))
  end
  esc(quote
    $(defs...)
  end)
end

@_import_io_iterators()
