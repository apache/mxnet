"""Root type for data provider

    A data provider provides interface to iterate over a dataset. It should implement the following functions:

      provides(provider :: AbstractDataProvider) => Vector{Tuple{Base.Symbol, Tuple}}

    Returns a list of name-shape pairs, indicating the name and shape of the each data stream. For example,
    `[(:data, (100,1,28,28)), (:softmax_label, (100,1))]`.

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

    A data batch must implement the following interface function to actually provide the data. The interface
    is designed to make it easy to generate data on the fly.

      load_data!(batch :: AbstractDataBatch, targets :: Dict{Base.Symbol, SlicedNDArray})

    Load data into targets. The target is a dictionary mapping name to actual `SlicedNDArray` the data should be
    copied into. Note `targets` might not contain names of all the data we could *provide*, simply because
    some the data we provie is not needed.

    The `SlicedNDArray` is used in data parallelization to run different sub-batch on different devices.
"""
abstract AbstractDataBatch


################################################################################
# MXDataProvider
################################################################################

"""Wrapper of built-in `libmxnet` data iterators.
"""
type MXDataProvider <: AbstractDataProvider
  handle   :: MX_DataIterHandle
  provides :: Vector{Tuple{Base.Symbol, Tuple}}
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
  provides = [(data_name, size(_get_data(handle)))]
  if !isa(label_name, Void)
    push!(provides, (label_name::Base.Symbol, size(_get_label(handle))))
  end

  MXDataProvider(handle, provides)
end

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
  return (MXDataBatch(provider.handle), state)
end

function load_data!(batch :: MXDataBatch, targets :: Dict{Base.Symbol, SlicedNDArray})
  for (k,v) in targets
    if k == batch.provider.provides[1][1]
      # data
      src = _get_data(batch.provider.handle)
    elseif k == batch.provider.provides[2][1]
      # label
      src = _get_label(batch.provider.handle)
    else
      @assert(false, "Unknown data $k, we only provide $(batch.provider.provides)")
    end

    for (idx, target) in v
      copy!(target, slice(src, idx))
    end
  end
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
  println("defining iterator $iter_name")
  defun = quote
    function $iter_name(; kwargs...)
      arg_keys = AbstractString[string(k) for (k,v) in kwargs]
      arg_vals = AbstractString[string(v) for (k,v) in kwargs]
      ref_hdr  = Ref{MX_handle}

      @mxcall(:MXDataIterCreateIter, (MX_handle, MX_uint, char_pp, char_pp, Ref{MX_handle}),
              $hdr, length(arg_keys), arg_keys, arg_vals, ref_hdr)

      return MXDataProvider(MX_DataIterHandle(ref_hdr[]); kwargs...)
    end
  end
  eval(defun)
  # TODO: add docstring
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
