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

@enum CONTEXT_TYPE CPU=1 GPU=2 CPU_PINNED=3

Base.convert(::Type{CONTEXT_TYPE}, x::Integer) = CONTEXT_TYPE(x)

"""
    Context(dev_type, dev_id)

A context describes the device type and id on which computation should be carried on.
"""
struct Context
  device_type::CONTEXT_TYPE
  device_id::Int

  Context(dev_type::CONTEXT_TYPE, dev_id::Integer = 0) = new(dev_type, dev_id)
end

const _default_ctx = Ref{Context}(Context(CPU, 0))

Context(dev_type::Integer, dev_id::Integer = 0) =
  Context(convert(CONTEXT_TYPE, dev_type), dev_id)

Base.show(io::IO, ctx::Context) =
  print(io, lowercase("$(ctx.device_type)$(ctx.device_id)"))

function _with_context(dev_type::Union{Symbol,Expr}, dev_id, e::Expr)
  global _default_ctx
  quote
    ctx = current_context()
    ctx′ = Context($(esc(dev_type)), $(esc(dev_id)))
    $_default_ctx[] = ctx′
    try
      return $(esc(e))
    finally
      $_default_ctx[] = ctx
    end
  end
end

"""
    @context device_type [device_id] expr

Change the default context in the following expression.

# Examples
```jl-repl
julia> mx.@context mx.GPU begin
         mx.zeros(2, 3)
       end
2×3 NDArray{Float32,2} @ gpu0:
 0.0f0  0.0f0  0.0f0
 0.0f0  0.0f0  0.0f0

julia> @context mx.GPU mx.zeros(3, 2)
3×2 NDArray{Float32,2} @ gpu0:
 0.0f0  0.0f0
 0.0f0  0.0f0
 0.0f0  0.0f0
```
"""
macro context(dev_type, e::Expr)
  _with_context(dev_type, 0, e)
end

macro context(dev_type, dev_id, e::Expr)
  _with_context(dev_type, dev_id, e)
end

for dev ∈ [:cpu, :gpu]
  ctx = QuoteNode(Symbol(uppercase(string(dev))))
  docstring = """
        @$dev [device_id] expr

    A shorthand for `@context mx.GPU`.

    # Examples
    ```jl-repl
    julia> mx.@with_gpu mx.zeros(2, 3)
    2×3 NDArray{Float32,2} @ gpu0:
     0.0f0  0.0f0  0.0f0
     0.0f0  0.0f0  0.0f0
    ```
    """
  @eval begin
    @doc $docstring ->
    macro $dev(e::Expr)
      ctx = $ctx
      quote
        @context $ctx $(esc(e))
      end
    end

    macro $dev(dev_id, e::Expr)
      ctx = $ctx
      quote
        @context $ctx $(esc(dev_id)) $(esc(e))
      end
    end
  end
end  # for dev ∈ [:cpu, :gpu]

"""
    cpu(dev_id)

Get a CPU context with a specific id. `cpu()` is usually the default context for many
operations when no context is specified.

# Arguments
* `dev_id::Integer = 0`: the CPU id.
"""
cpu(dev_id::Integer = 0) = Context(CPU, dev_id)

"""
    gpu(dev_id)

Get a GPU context with a specific id. The K GPUs on a node is typically numbered as 0,...,K-1.

# Arguments
* `dev_id::Integer = 0` the GPU device id.
"""
gpu(dev_id::Integer = 0) = Context(GPU, dev_id)

"""
    num_gpus()

Query CUDA for the number of GPUs present.
"""
function num_gpus()
  n = Ref{Cint}()
  @mxcall :MXGetGPUCount (Ref{Cint},) n
  n[]
end

"""
    empty_cache(ctx::Context = current_context())

Empties the memory cache for the current contexts device.
MXNet utilizes a memory pool to avoid excessive allocations.
Calling empty_cache will empty the memory pool of the contexts
device. This will only free the memory of the unreferenced data.
"""
function empty_cache(ctx::Context = current_context())
  @mxcall :MXStorageEmptyCache (Cint, Cint) ctx.device_type ctx.device_id
  ctx
end

"""
    gpu_memory_info(dev_id = 0)::Tuple{UInt64,UInt64}

Query CUDA for the free and total bytes of GPU global memory.
It returns a tuple of `(free memory, total memory)`.

```julia-repl
julia> mx.gpu_memory_info()
(0x00000003af240000, 0x00000003f9440000)
```
"""
function gpu_memory_info(dev_id = 0)
  free = Ref{UInt64}()
  n = Ref{UInt64}()
  @mxcall :MXGetGPUMemoryInformation64 (Cint, Ref{UInt64}, Ref{UInt64}) dev_id free n
  free[], n[]
end

"""
    current_context()

Return the current context.

By default, `mx.cpu()` is used for all the computations
and it can be overridden by using the `@context` macro.

# Examples
```jl-repl
julia> mx.current_context()
cpu0

julia> mx.@context mx.GPU 1 begin  # Context changed in the following code block
         mx.current_context()
       end
gpu1

julia> mx.current_context()
cpu0
```
"""
current_context() = _default_ctx[]
