"""Root type for data provider

A data provider provides interface to iterate over a dataset. It should implement the following functions:

```julia
provide_data(provider :: AbstractDataProvider) => Vector{Tuple{Base.Symbol, Tuple}}
provide_label(provider :: AbstractDataProvider) => Vector{Tuple{Base.Symbol, Tuple}}
```

Returns a list of name-shape pairs, indicating the name and shape of the each data stream. For example,
`[(:data, (100,1,28,28))]` or `[(:softmax_label, (100,1))]`. It should also implement the following convenient
function

```julia
get_batch_size(provider :: AbstractDataProvider) => Int
```

which returns the batch size used in this data provider.

A data provider should implement the standard Julia iteration interface, including `Base.start`,
`Base.next`, `Base.done` and `Base.eltype`. It could safely assume that the interface functions will
always be called like

```julia
for batch in provider
  # ...
  load_data!(batch, targets)
end
```

which translates into

```julia
state = Base.start(provider)
while !Base.done(provider, state)
  (batch, state) = Base.next(provider, state)
  # ...
  load_data!(batch, targets)
end
```

In other words, it could safely assume that `Base.next` is always called after `Base.done`. And neither
of those function will be called twice consequtively. The detailed interfaces are list below:

```julia
Base.start(provider :: AbstractDataProvider) => AbstractDataProviderState
```

Initialize or reset the data iteration.

```julia
Base.next(provider :: AbstractDataProvider, state :: AbstractDataProviderState)
    => (AbstractDataBatch, AbstractDataProviderState)
```

Return one batch of data. Actual data can be retrieved from the batch by interface functions described
in the document of type `AbstractDataBatch`.

```julia
Base.done(provider :: AbstractDataProvider, state :: AbstractDataProviderState) => Bool
```

Return `false` if there is more batch to get.

```julia
Base.eltype(::Type{MyDataProvider}) => MyDataProviderState
```

Return the type of the data provider state.
"""
abstract AbstractDataProvider

"""Root type for states of data provider"""
abstract AbstractDataProviderState

"""A tuple of (slice, NDArray). Usually each NDArray resides on a different device, and each
    slice describe which part of a larger piece of data should goto that device.
"""
typealias SlicedNDArray Tuple{UnitRange{Int},NDArray}

"""Root type for data batch

A data batch must implement the following interface function to actually provide the data and label.

```julia
load_data!(batch :: AbstractDataBatch, targets :: Vector{Vector{SlicedNDArray}})
load_label!(batch :: AbstractDataBatch, targets :: Vector{Vector{SlicedNDArray}})
```

Load data and label into targets. The targets is a list of target that the data/label should be
copied into. The order in the list is guaranteed to be the same as returned by `provide_data` and
`provide_label`. Each entry in the list is again a list of `SlicedNDArray`, corresponding the
memory buffer for each device.

The `SlicedNDArray` is used in data parallelization to run different sub-batch on different devices.

The following function should also be implemented to handle the case when the mini-batch size does not
divide the size of the whole dataset. So in the last mini-batch, the actual data copied might be fewer
than the mini-batch size. This is usually not an issue during the training as the remaining space may
contain the data and label copied during the previous mini-batch are still valid data. However, during
testing, especially when doing feature extraction, we need to be precise about the number of samples
processed.

```julia
get_pad(batch :: AbstractDataBatch)
```

Return the number of *dummy samples* in this mini-batch.
"""
abstract AbstractDataBatch


################################################################################
# ArrayDataProvider
################################################################################
"A convenient tool to iterate `NDArray` or Julia `Array`"
type ArrayDataProvider <: AbstractDataProvider
  data_arrays :: Vector{Array{MX_float}}
  data_names  :: Vector{Base.Symbol}
  label_arrays:: Vector{Array{MX_float}}
  label_names :: Vector{Base.Symbol}
  batch_size  :: Int
  sample_count:: Int
  shuffle     :: Bool
end


# Julia's type system is sometimes very frustrating. You cannot specify a function
# with argument Vector{Pair} to expect to be matched when calling with the parameter
# [:foo => zeros(2,3), :bar => zeros(3)] because the type inference gives very specific
# results, about the parametric type in the Pair{T1,T2} type, thus does not match the
# generic Pair type. In general, Int <: Number but Vector{Int} <: Vector{Number} is not
# true. So let us just use Any here...
function ArrayDataProvider(data::Any; batch_size::Int=1, shuffle::Bool=false)
  ArrayDataProvider(data, [], batch_size=batch_size, shuffle=shuffle)
end
function ArrayDataProvider(data::Any, label::Any; batch_size::Int=1, shuffle::Bool=false)
  if isa(data, Union{NDArray, Array}) && eltype(data) <: Real
    data_names  = [:data]
    data_arrays = Array{MX_float}[data]
  elseif isa(data, Pair)
    @assert isa(data.first, Base.Symbol) && isa(data.second, Union{NDArray, Array})
    data_names  = [data.first]
    data_arrays = Array{MX_float}[data.second]
  elseif isa(data, Vector) || isa(data, Tuple)
    map(data) do d
      @assert isa(d, Pair) && isa(d.first, Base.Symbol) && isa(d.second, Union{NDArray, Array})
    end
    data_names  = Base.Symbol[d.first for d in data]
    data_arrays = Array{MX_float}[d.second for d in data]
  else
    error("Invalid data argument type")
  end

  if isa(label, Union{NDArray, Array}) && eltype(label) <: Real
    label_names  = [:softmax_label]
    label_arrays = Array{MX_float}[label]
  elseif isa(label, Pair)
    @assert isa(label.first, Base.Symbol) && isa(label.second, Union{NDArray, Array})
    label_names  = [label.first]
    label_arrays = Array{MX_float}[label.second]
  elseif isa(label, Vector) || isa(label, Tuple)
    map(label) do d
      @assert isa(d, Pair) && isa(d.first, Base.Symbol) && isa(d.second, Union{NDArray, Array})
    end
    label_names  = Base.Symbol[d.first for d in label]
    label_arrays = Array{MX_float}[d.second for d in label]
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

  ArrayDataProvider(data_arrays, data_names, label_arrays, label_names, batch_size, sample_count, shuffle)
end

function provide_data(provider::ArrayDataProvider)
  return collect(zip(provider.data_names, map(size, provider.data_arrays)))
end
function provide_label(provider::ArrayDataProvider)
  return collect(zip(provider.label_names, map(size, provider.label_arrays)))
end
get_batch_size(provider::ArrayDataProvider) = provider.batch_size

immutable ArrayDataProviderState <: AbstractDataProviderState
  curr_idx :: Int
end

function Base.eltype(provider :: ArrayDataProvider)
  ArrayDataProviderState
end

function _shuffle_array(arr::Array, idx::Vector{Int})
  shape  = size(arr)
  colons = [Colon() for c = 1:length(shape)-1]
  getindex(arr, colons..., idx)
end
function Base.start(provider :: ArrayDataProvider)
  if provider.shuffle
    # re-shuffle all data
    idx_perm = randperm(provider.sample_count)
    provider.data_arrays = map(x->_shuffle_array(x,idx_perm), provider.data_arrays)
    provider.label_arrays = map(x->_shuffle_array(x,idx_perm), provider.label_arrays)
  end

  return ArrayDataProviderState(1)
end

function Base.done(provider::ArrayDataProvider, state :: ArrayDataProviderState)
  return state.curr_idx > provider.sample_count
end

immutable ArrayDataBatch <: AbstractDataBatch
  provider :: ArrayDataProvider
  idx      :: UnitRange{Int}
end
function Base.next(provider :: ArrayDataProvider, state :: ArrayDataProviderState)
  idx = state.curr_idx:min(state.curr_idx+provider.batch_size, provider.sample_count)
  return (ArrayDataBatch(provider, idx), ArrayDataProviderState(idx.stop+1))
end

function get_pad(batch :: ArrayDataBatch)
  return batch.provider.batch_size - length(batch.idx)
end

function _load_general!(batch :: ArrayDataBatch, sources :: Vector{Array{MX_float}},
                        targets :: Vector{Vector{SlicedNDArray}})
  @assert length(sources) == length(targets)
  for (src, tgt) in zip(sources, targets)
    src_colons = [Colon() for i = 1:ndims(src)-1]
    for (slice_idx, dst) in tgt
      copy!(dst, getindex(src, src_colons..., batch.idx[slice_idx]))
    end
  end
end
function load_data!(batch :: ArrayDataBatch, targets :: Vector{Vector{SlicedNDArray}})
  _load_general!(batch, batch.provider.data_arrays, targets)
end
function load_label!(batch :: ArrayDataBatch, targets :: Vector{Vector{SlicedNDArray}})
  _load_general!(batch, batch.provider.label_arrays, targets)
end



################################################################################
# MXDataProvider
################################################################################

"""Wrapper of built-in `libmxnet` data iterators.
"""
type MXDataProvider <: AbstractDataProvider
  handle     :: MX_DataIterHandle
  data_shape :: Vector{Tuple{Base.Symbol, Tuple}}
  label_shape:: Vector{Tuple{Base.Symbol, Tuple}}
  batch_size :: Int
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
                        data_name  :: Union{Base.Symbol,Void}=:data,
                        label_name :: Union{Base.Symbol,Void}=:softmax_label,
                        kwargs...) # for convenience, we ignore the rest keyword arguments
  # init iterator, load the first batch and get shapes
  _reset_data_iter(handle)
  @assert(_iter_next(handle), "Failed to load the first batch in MXDataProvider")
  data_shape = Tuple{Base.Symbol, Tuple}[(data_name, size(_get_data(handle)))]
  if !isa(label_name, Void)
    label_shape = Tuple{Base.Symbol, Tuple}[(label_name::Base.Symbol, size(_get_label(handle)))]
  else
    label_shape = Tuple{Base.Symbol, Tuple}[]
  end
  _reset_data_iter(handle)

  MXDataProvider(handle, data_shape, label_shape, data_shape[1][2][end])
end

provide_data(provider::MXDataProvider) = provider.data_shape
provide_label(provider::MXDataProvider) = provider.label_shape
get_batch_size(provider::MXDataProvider) = provider.batch_size

type MXDataProviderState <: AbstractDataProviderState
  has_next :: Bool
end
type MXDataBatch <: AbstractDataBatch
  provider :: MXDataProvider
end

function Base.eltype(provider :: MXDataProvider)
  MXDataBatch
end
function Base.start(provider :: MXDataProvider)
  _reset_data_iter(provider.handle)
  return MXDataProviderState(true)
end
function Base.done(provider :: MXDataProvider, state :: MXDataProviderState)
  state.has_next = _iter_next(provider.handle)
  return !state.has_next
end
function Base.next(provider :: MXDataProvider, state :: MXDataProviderState)
  return (MXDataBatch(provider), state)
end

function _load_general!(batch :: MXDataBatch, loader :: Function, targets :: Vector{Vector{SlicedNDArray}})
  @assert length(targets) == 1
  src = loader(batch.provider.handle)
  for (idx, target) in targets[1]
    copy!(target, slice(src, idx))
  end
end

function load_data!(batch :: MXDataBatch, targets :: Vector{Vector{SlicedNDArray}})
  _load_general!(batch, _get_data, targets)
end
function load_label!(batch :: MXDataBatch, targets :: Vector{Vector{SlicedNDArray}})
  _load_general!(batch, _get_label, targets)
end

function get_pad(batch :: MXDataBatch)
  ref_pad = Ref{Cint}(0)
  @mxcall(:MXDataIterGetPadNum, (MX_handle, Ref{Cint}), batch.provider.handle, ref_pad)
  return Int(ref_pad[])
end


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

  iter_name = symbol(bytestring(ref_name[]))
  defun = quote
    function $iter_name(; kwargs...)
      arg_keys = AbstractString[string(k) for (k,v) in kwargs]
      arg_vals = AbstractString[dump_mx_param(v) for (k,v) in kwargs]
      ref_hdr  = Ref{MX_handle}(0)

      @mxcall(:MXDataIterCreateIter, (MX_handle, MX_uint, char_pp, char_pp, Ref{MX_handle}),
              $hdr, length(arg_keys), arg_keys, arg_vals, ref_hdr)

      return MXDataProvider(MX_DataIterHandle(ref_hdr[]); kwargs...)
    end
  end
  eval(defun)
  # TODO: add docstring

  # add an alias XXXProvider => XXXIter
  if endswith(string(iter_name), "Iter")
    alias_name = symbol(string(iter_name)[1:end-4] * "Provider")
    eval(:($alias_name = $iter_name))
  end
end

function _import_io_iterators()
  n_ref = Ref{MX_uint}(0)
  h_ref = Ref{Ptr{MX_handle}}(0)
  @mxcall(:MXListDataIters, (Ref{MX_uint}, Ref{Ptr{MX_handle}}), n_ref, h_ref)

  n_creators = n_ref[]
  h_creators = pointer_to_array(h_ref[], n_creators)

  for i = 1:n_creators
    creator_hdr = h_creators[i]
    _define_data_iter_creator(creator_hdr)
  end
end
