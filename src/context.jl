@enum CONTEXT_TYPE CPU=1 GPU=2 CPU_PINNED=3

"""
    Context(dev_type, dev_id)

A context describes the device type and id on which computation should be carried on.
"""
struct Context
  device_type :: CONTEXT_TYPE
  device_id   :: Int
end
Context(dev_type :: Union{CONTEXT_TYPE, Int}, dev_id :: Int = 0) =
    Context(convert(CONTEXT_TYPE, dev_type), dev_id)

function Base.show(io :: IO, ctx :: Context)
  print(io, "$(ctx.device_type)$(ctx.device_id)")
end

"""
    cpu(dev_id)

Get a CPU context with a specific id. `cpu()` is usually the default context for many
operations when no context is specified.

# Arguments
* `dev_id::Int = 0`: the CPU id.
"""
function cpu(dev_id::Int=0)
  return Context(CPU, dev_id)
end

"""
    gpu(dev_id)

Get a GPU context with a specific id. The K GPUs on a node is typically numbered as 0,...,K-1.

# Arguments
* `dev_id :: Int = 0` the GPU device id.
"""
function gpu(dev_id::Int=0)
  return Context(GPU, dev_id)
end
