"""Root type for data provider

    A data provider provides interface to iterate over a dataset. It should implement the following functions:

      provide_data(provider :: AbstractDataProvider) => Vector{Tuple{Base.Symbol, Tuple}}
      provide_label(provider :: AbstractDataProvider) => Vector{Tuple{Base.Symbol, Tuple}}

    Returns a list of name-shape pairs, indicating the name and shape of the each data stream. For example,
    `[(:data, (100,1,28,28))]` or `[(:softmax_label, (100,1))]`. It should also implement the following convenient
    function

      get_batch_size(provider :: AbstractDataProvider) => Int

    which returns the batch size used in this data provider.

    A data provider should implement the standard Julia iteration interface, including `Base.start`,
    `Base.next`, `Base.done` and `Base.eltype`. It could safely assume that the interface functions will
    always be called like

      for batch in provider
        # ...
        load_data!(batch, targets)
      end

    which translates into

      state = Base.start(provider)
      while !done(provider, state)
        (batch, state) = next(provider, state)
        # ...
        load_data!(batch, targets)
      end

    In other words, it could safely assume that `Base.next` is always called after `Base.done`. And neither
    of those function will be called twice consequtively. The detailed interfaces are list below:

      Base.start(provider :: AbstractDataProvider) => AbstractDataProviderState

    Initialize or reset the data iteration.

      Base.next(provider :: AbstractDataProvider, state :: AbstractDataProviderState)
          => (AbstractDataBatch, AbstractDataProviderState)

    Return one batch of data. Actual data can be retrieved from the batch by interface functions described
    in the document of type `AbstractDataBatch`.

      Base.done(provider :: AbstractDataProvider, state :: AbstractDataProviderState) => Bool

    Return `false` if there is more batch to get.

      Base.eltype(::Type{MyDataProvider}) => MyDataProviderState

    Return the type of the data provider state.
"""
abstract AbstractDataProvider

"""Root type for states of data provider"""
abstract AbstractDataProviderState

"""A list of (slice, NDArray) pairs. Usually each NDArray resides on a different device, and each
    slice describe which part of a larger piece of data should goto that device.
"""
typealias SlicedNDArray Vector{Tuple{UnitRange{Int},NDArray}}

"""Root type for data batch

    A data batch must implement the following interface function to actually provide the data and label.

      load_data!(batch :: AbstractDataBatch, targets :: Vector{SlicedNDArray})
      load_label!(batch :: AbstractDataBatch, targets :: Vector{SlicedNDArray})

    Load data and label into targets. The target is a list of `SlicedNDArray` the data/label should be
    copied into. The order in the list is guaranteed to be the same as returned by `provide_data` and
    `provide_label`.

    The `SlicedNDArray` is used in data parallelization to run different sub-batch on different devices.

    The following function should also be implemented to handle the case when the mini-batch size does not
    divide the size of the whole dataset. So in the last mini-batch, the actual data copied might be fewer
    than the mini-batch size. This is usually not an issue during the training as the remaining space may
    contain the data and label copied during the previous mini-batch are still valid data. However, during
    testing, especially when doing feature extraction, we need to be precise about the number of samples
    processed.

      get_pad(batch :: AbstractDataBatch)

    Return the number of *dummy samples* in this mini-batch.
"""
abstract AbstractDataBatch


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

function _load_general!(batch :: MXDataBatch, loader :: Function, targets :: Vector{SlicedNDArray})
  @assert length(targets) == 1
  src = loader(batch.provider.handle)
  for (idx, target) in targets[1]
    copy!(target, slice(src, idx))
  end
end

function load_data!(batch :: MXDataBatch, targets :: Vector{SlicedNDArray})
  _load_general!(batch, _get_data, targets)
end
function load_label!(batch :: MXDataBatch, targets :: Vector{SlicedNDArray})
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
      arg_vals = AbstractString[string(v) for (k,v) in kwargs]
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
