#=doc
Data Providers
==============

Data providers are wrappers that load external data, be it images, text, or general tensors,
and split it into mini-batches so that the model can consume the data in a uniformed way.
=#

#=doc
.. class:: AbstractDataProvider

   The root type for all data provider. A data provider should implement the following interfaces:

   .. function:: get_batch_size(provider) -> Int

      :param AbstractDataProvider provider: the data provider.
      :return: the mini-batch size of the provided data. All the provided data should have the
               same mini-batch size (i.e. the last dimension).

   .. function:: provide_data(provider) -> Vector{Tuple{Base.Symbol, Tuple}}

      :param AbstractDataProvider provider: the data provider.
      :return: a vector of (name, shape) pairs describing the names of the data it provides, and
               the corresponding shapes.

   .. function:: provide_label(provider) -> Vector{Tuple{Base.Symbol, Tuple}}

      :param AbstractDataProvider provider: the data provider.
      :return: a vector of (name, shape) pairs describing the names of the labels it provides, and
               the corresponding shapes.

   The difference between *data* and *label* is that during
   training stage, both *data* and *label* will be feeded into the model, while during
   prediction stage, only *data* is loaded. Otherwise, they could be anything, with any names, and
   of any shapes. The provided data and label names here should match the input names in a target
   :class:`Symbol`.

   A data provider should also implement the Julia iteration interface, in order to allow iterating
   through the data set. The provider will be called in the following way:

   .. code-block:: julia

      for batch in provider
        data = get_data(provider, batch)
      end

   which will be translated by Julia compiler into

   .. code-block:: julia

      state = Base.start(provider)
      while !Base.done(provider, state)
        (batch, state) = Base.next(provider, state)
        data = get_data(provider, batch)
      end

   The detailed interface function is listed below:

   .. function:: Base.eltype(provider) -> AbstractDataBatch

      :param AbstractDataProvider provider: the data provider.
      :return: the specific subtype representing a data batch. See :class:`AbstractDataBatch`.

   .. function:: Base.start(provider) -> AbstractDataProviderState

      :param AbstractDataProvider provider: the data provider.

      This function is always called before iterating into the dataset. It should initialize
      the iterator, reset the index, and do data shuffling if needed.

   .. function:: Base.done(provider, state) -> Bool

      :param AbstractDataProvider provider: the data provider.
      :param AbstractDataProviderState state: the state returned by :func:`Base.start` :func:`Base.next`.
      :return: true if there is no more data to iterate in this dataset.

   .. function:: Base.next(provider) -> (AbstractDataBatch, AbstractDataProviderState)

      :param AbstractDataProvider provider: the data provider.
      :return: the current data batch, and the state for the next iteration.

   Note sometimes you are wrapping an existing data iterator (e.g. the built-in libmxnet data iterator) that
   is built with a different convention. It might be difficult to adapt to the interfaces stated here. In this
   case, you can safely assume that

   * :func:`Base.start` will always be called, and called only once before the iteration starts.
   * :func:`Base.done` will always be called at the beginning of every iteration and always be called once.
   * If :func:`Base.done` return true, the iteration will stop, until the next round, again, starting with
     a call to :func:`Base.start`.
   * :func:`Base.next` will always be called only once in each iteration. It will always be called after
     one and only one call to :func:`Base.done`; but if :func:`Base.done` returns true, :func:`Base.next` will
     not be called.

   With those assumptions, it will be relatively easy to adapt any existing iterator. See the implementation
   of the built-in :class:`MXDataProvider` for example.
=#
abstract AbstractDataProvider

#=doc
.. class:: AbstractDataProviderState

   Base type for data provider states.
=#
abstract AbstractDataProviderState

#=doc
.. class:: AbstractDataBatch

   Base type for a data mini-batch. It should implement the following interfaces:

   .. function:: count_samples(provider, batch) -> Int

      :param AbstractDataBatch batch: the data batch object.
      :return: the number of samples in this batch. This number should be greater than 0, but
               less than or equal to the batch size. This is used to indicate at the end of
               the data set, there might not be enough samples for a whole mini-batch.

   .. function:: get_data(provider, batch) -> Vector{NDArray}

      :param AbstractDataProvider provider: the data provider.
      :param AbstractDataBatch batch: the data batch object.
      :return: a vector of data in this batch, should be in the same order as declared in
               :func:`provide_data() <AbstractDataProvider.provide_data>`.

               The last dimension of each :class:`NDArray` should always match the batch_size, even when
               :func:`count_samples` returns a value less than the batch size. In this case,
               the data provider is free to pad the remaining contents with any value.

   .. function:: get_label(provider, batch) -> Vector{NDArray}

      :param AbstractDataProvider provider: the data provider.
      :param AbstractDataBatch batch: the data batch object.
      :return: a vector of labels in this batch. Similar to :func:`get_data`.


   The following utility functions will be automatically defined.

   .. function:: get(provider, batch, name) -> NDArray

      :param AbstractDataProvider provider: the data provider.
      :param AbstractDataBatch batch: the data batch object.
      :param Base.Symbol name: the name of the data to get, should be one of the names
             provided in either :func:`provide_data() <AbstractDataProvider.provide_data>`
             or :func:`provide_label() <AbstractDataprovider.provide_label>`.
      :return: the corresponding data array corresponding to that name.

   .. function:: load_data!(provider, batch, targets)

      :param AbstractDataProvider provider: the data provider.
      :param AbstractDataBatch batch: the data batch object.
      :param targets: the targets to load data into.
      :type targets: Vector{Vector{SlicedNDArray}}

      The targets is a list of the same length as number of data provided by this provider.
      Each element in the list is a list of :class:`SlicedNDArray`. This list described a
      spliting scheme of this data batch into different slices, each slice is specified by
      a slice-ndarray pair, where *slice* specify the range of samples in the mini-batch
      that should be loaded into the corresponding *ndarray*.

      This utility function is used in data parallelization, where a mini-batch is splited
      and computed on several different devices.

   .. function:: load_label!(provider, batch, targets)

      :param AbstractDataProvider provider: the data provider.
      :param AbstractDataBatch batch: the data batch object.
      :param targets: the targets to load label into.
      :type targets: Vector{Vector{SlicedNDArray}}

      The same as :func:`load_data!`, except that this is for loading labels.
=#
abstract AbstractDataBatch

#=doc
.. class:: SlicedNDArray

   A alias type of ``Tuple{UnitRange{Int},NDArray}``.
=#
typealias SlicedNDArray Tuple{UnitRange{Int},NDArray}

function _load_general!(provider :: AbstractDataProvider, batch :: AbstractDataBatch,
                        targets :: Vector{Vector{SlicedNDArray}}, loader::Function)
  data = loader(provider, batch)
  for (d_src, d_targets) in zip(data, targets)
    for (slice_idx, d_dst) in d_targets
      copy!(d_dst, slice(d_src, slice_idx))
    end
  end
end
function load_data!(provider :: AbstractDataProvider, batch :: AbstractDataBatch,
                    targets :: Vector{Vector{SlicedNDArray}})
  _load_general!(provider, batch, targets, get_data)
end
function load_label!(provider :: AbstractDataProvider, batch :: AbstractDataBatch,
                     targets :: Vector{Vector{SlicedNDArray}})
  _load_general!(provider, batch, targets, get_label)
end

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

The Batch type should have a field named `provider` pointing to the underlying provider. Helper functions
`get_data` and `get_label` (mainly for debug purpose) will be able to use this.
"""

#function _get_data_or_label(batch::AbstractDataBatch, provide_func::Function, loader::Function)
#  data_shapes = provide_func(batch.provider)
#  data_arrays = [mx.empty(x[2]) for x in data_shapes]
#  batch_size  = get_batch_size(batch.provider)
#  data_arrays_fake_slice = [SlicedNDArray[(1:batch_size, x)] for x in data_arrays]
#  loader(batch, data_arrays_fake_slice)
#
#  if length(data_arrays) == 1
#    return data_arrays[1]
#  else
#    return data_arrays
#  end
#end
#function get_data(batch :: AbstractDataBatch)
#  _get_data_or_label(batch, provide_data, load_data!)
#end
#function get_label(batch :: AbstractDataBatch)
#  _get_data_or_label(batch, provide_label, load_label!)
#end

################################################################################
# ArrayDataProvider
################################################################################
"A convenient tool to iterate `NDArray` or Julia `Array`"
type ArrayDataProvider <: AbstractDataProvider
  data_arrays   :: Vector{Array{MX_float}}
  data_names    :: Vector{Base.Symbol}
  label_arrays  :: Vector{Array{MX_float}}
  label_names   :: Vector{Base.Symbol}
  batch_size    :: Int
  sample_count  :: Int
  shuffle       :: Bool
  data_padding  :: MX_float
  label_padding :: MX_float
end


# Julia's type system is sometimes very frustrating. You cannot specify a function
# with argument Vector{Pair} to expect to be matched when calling with the parameter
# [:foo => zeros(2,3), :bar => zeros(3)] because the type inference gives very specific
# results, about the parametric type in the Pair{T1,T2} type, thus does not match the
# generic Pair type. In general, Int <: Number but Vector{Int} <: Vector{Number} is not
# true. So let us just use Any here...
function ArrayDataProvider(data::Any; batch_size::Int=1, shuffle::Bool=false, data_padding::Real=0, label_padding::Real=0)
  ArrayDataProvider(data, [], batch_size=batch_size, shuffle=shuffle, data_padding=data_padding, label_padding=label_padding)
end
function ArrayDataProvider(data::Any, label::Any; batch_size::Int=1, shuffle::Bool=false, data_padding::Real=0, label_padding::Real=0)
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

  ArrayDataProvider(data_arrays, data_names, label_arrays, label_names, batch_size,
                    sample_count, shuffle, data_padding, label_padding)
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
  idx = state.curr_idx:min(state.curr_idx+provider.batch_size-1, provider.sample_count)
  return (ArrayDataBatch(provider, idx), ArrayDataProviderState(idx.stop+1))
end

function get_pad(batch :: ArrayDataBatch)
  return batch.provider.batch_size - length(batch.idx)
end

function _load_general!(batch :: ArrayDataBatch, sources :: Vector{Array{MX_float}},
                        targets :: Vector{Vector{SlicedNDArray}}, pad_val::Real)
  @assert length(sources) == length(targets)
  for (src, tgt) in zip(sources, targets)
    src_colons = [Colon() for i = 1:ndims(src)-1]
    for (slice_idx, dst) in tgt
      if slice_idx.start > length(batch.idx)
        dst[:] = pad_val
      else
        slice_idx0 = slice_idx.start:min(slice_idx.stop, length(batch.idx))
        copy!(dst[1:length(slice_idx0)], getindex(src, src_colons..., batch.idx[slice_idx0]))
        if length(slice_idx0) < length(slice_idx)
          # need padding
          dst[length(slice_idx0)+1:length(slice_idx)] = pad_val
        end
      end
    end
  end
end
function load_data!(batch :: ArrayDataBatch, targets :: Vector{Vector{SlicedNDArray}})
  _load_general!(batch, batch.provider.data_arrays, targets, batch.provider.data_padding)
end
function load_label!(batch :: ArrayDataBatch, targets :: Vector{Vector{SlicedNDArray}})
  _load_general!(batch, batch.provider.label_arrays, targets, batch.provider.label_padding)
end



################################################################################
#=doc
.. class:: MXDataProvider

   A data provider that wrap built-in data iterators from libmxnet.
=#
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
                        data_name  :: Base.Symbol=:data,
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

  MXDataProvider(handle, data_shape, label_shape, data_shape[1][2][end])
end

provide_data(provider::MXDataProvider) = provider.data_shape
provide_label(provider::MXDataProvider) = provider.label_shape
get_batch_size(provider::MXDataProvider) = provider.batch_size

type MXDataProviderState <: AbstractDataProviderState
  has_next :: Bool
end
immutable MXDataBatch <: AbstractDataBatch
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
  @mxcall(:MXDataIterGetPadNum, (MX_handle, Ref{Cint}), batch.provider.handle, ref_pad)
  return provider.batch_size - Int(ref_pad[])
end

#=doc
Built-in data providers in libmxnet
-----------------------------------

**autogen:EMBED:io:EMBED:autogen**
=#
function _define_data_iter_creator(hdr :: MX_handle; gen_docs::Bool=false)
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

  if gen_docs
    if endswith(string(iter_name), "Iter")
      f_desc = "Can also be called with the alias ``$(string(iter_name)[1:end-4] * "Provider")``.\n"
    else
      f_desc = ""
    end
    f_desc *= bytestring(ref_desc[]) * "\n\n"
    f_desc *= ":param Base.Symbol data_name: keyword argument, default ``:data``. The name of the data.\n"
    f_desc *= ":param Base.Symbol label_name: keyword argument, default ``:softmax_label``. " *
              "The name of the label. Could be ``nothing`` if no label is presented in this dataset.\n\n"
    f_desc *= _format_docstring(Int(ref_narg[]), ref_arg_names, ref_arg_types, ref_arg_descs)
    f_desc *= ":return: the constructed :class:`MXDataProvider`."
    return (iter_name, f_desc)
  end

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

  # add an alias XXXProvider => XXXIter
  if endswith(string(iter_name), "Iter")
    alias_name = symbol(string(iter_name)[1:end-4] * "Provider")
    eval(:($alias_name = $iter_name))
  end
end

function _import_io_iterators(;gen_docs::Bool=false)
  n_ref = Ref{MX_uint}(0)
  h_ref = Ref{Ptr{MX_handle}}(0)
  @mxcall(:MXListDataIters, (Ref{MX_uint}, Ref{Ptr{MX_handle}}), n_ref, h_ref)

  n_creators = n_ref[]
  h_creators = pointer_to_array(h_ref[], n_creators)

  if gen_docs
    docs = Dict{Base.Symbol, AbstractString}()
  end

  for i = 1:n_creators
    creator_hdr = h_creators[i]
    ret = _define_data_iter_creator(creator_hdr; gen_docs=gen_docs)
    if gen_docs
      docs[ret[1]] = ret[2]
    end
  end

  if gen_docs
    return docs
  end
end
