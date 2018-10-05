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

Base.show(io::IO, ctx::Context) =
  print(io, "$(ctx.device_type)$(ctx.device_id)")

"""
    cpu(dev_id)

Get a CPU context with a specific id. `cpu()` is usually the default context for many
operations when no context is specified.

# Arguments
* `dev_id::Int = 0`: the CPU id.
"""
cpu(dev_id::Int = 0) = Context(CPU, dev_id)

"""
    gpu(dev_id)

Get a GPU context with a specific id. The K GPUs on a node is typically numbered as 0,...,K-1.

# Arguments
* `dev_id :: Int = 0` the GPU device id.
"""
gpu(dev_id::Int = 0) = return Context(GPU, dev_id)
