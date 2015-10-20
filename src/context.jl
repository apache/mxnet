@enum CONTEXT_TYPE CPU=1 GPU=2 CPU_PINNED=3

immutable Context
  device_type :: CONTEXT_TYPE
  device_id   :: Int
end
Context(dev_type :: Union{CONTEXT_TYPE, Int}, dev_id :: Int = 0) =
    Context(convert(CONTEXT_TYPE, dev_type), dev_id)

function Base.show(io :: IO, ctx :: Context)
  print(io, "$(ctx.device_type)$(ctx.device_id)")
end

function cpu(dev_id::Int=0)
  return Context(CPU, dev_id)
end
function gpu(dev_id::Int=0)
  return Context(GPU, dev_id)
end
