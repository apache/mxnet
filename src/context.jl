@enum CONTEXT_TYPE CPU=1 GPU=2

type Context
  device_type :: CONTEXT_TYPE
  device_id   :: Cint

  old_ctx     :: Nullable{Context}
end
Context(dev_type :: CONTEXT_TYPE, dev_id = 0) =
    Context(dev_type, dev_id, Nullable{Context}())


# global default context
DEFAULT_CONTEXT = Context(CPU)
