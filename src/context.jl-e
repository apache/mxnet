#=doc
Context
=======
=#
@enum CONTEXT_TYPE CPU=1 GPU=2 CPU_PINNED=3

#=doc
.. class:: Context

   A context describes the device type and id on which computation should be carried on.
=#
immutable Context
  device_type :: CONTEXT_TYPE
  device_id   :: Int
end
Context(dev_type :: Union{CONTEXT_TYPE, Int}, dev_id :: Int = 0) =
    Context(convert(CONTEXT_TYPE, dev_type), dev_id)

function Base.show(io :: IO, ctx :: Context)
  print(io, "$(ctx.device_type)$(ctx.device_id)")
end

#=doc
.. function:: cpu(dev_id=0)

   :param Int dev_id: the CPU id.

   Get a CPU context with a specific id. ``cpu()`` is usually the default context for many
   operations when no context is specified.
=#
function cpu(dev_id::Int=0)
  return Context(CPU, dev_id)
end

#=doc
.. function:: gpu(dev_id=0)

   :param Int dev_id: the GPU device id.

   Get a GPU context with a specific id. The K GPUs on a node is typically numbered as 0,...,K-1.
=#
function gpu(dev_id::Int=0)
  return Context(GPU, dev_id)
end
