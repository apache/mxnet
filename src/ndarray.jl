export NDArray
export delete

function _ndarray_alloc{N}(shape :: NTuple{N, Int}, ctx :: Context, delay_alloc :: Bool)
  h_ref  = Ref{Ptr{Void}}(0)
  shape  = MX_uint[shape...]
  @mxcall(:MXNDArrayCreate, (Ptr{MX_uint}, MX_uint, Cint, Cint, Cint, Ref{Ptr{Void}}),
      shape, length(shape), ctx.device_type, ctx.device_id, delay_alloc, h_ref)
  handle = MX_NDArrayHandle(h_ref[])
  return handle
end

type NDArray
  handle   :: MX_NDArrayHandle
  writable :: Bool

  function NDArray(handle, writable=true)
    obj = new(handle, writable)

    # TODO: there is currently no good way of automatically managing external resources
    # using finalizers is said to slow down the GC significantly
    #finalizer(obj, delete)
    obj
  end
end

#function delete(obj :: NDArray)
#  if !isnull(obj.handle)
#    @mxcall(:MXNDArrayFree, (Ptr{Void},), obj.handle)
#    reset(obj.handle)
#  end
#end

function empty{N}(shape :: NTuple{N, Int}, ctx :: Context = DEFAULT_CONTEXT)
  NDArray(_ndarray_alloc(shape, ctx, false))
end
function empty(shape :: Int...)
  empty(shape)
end

