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

struct NDArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end
NDArrayStyle(::Val{N}) where N        = NDArrayStyle{N}()
NDArrayStyle{M}(::Val{N}) where {N,M} = NDArrayStyle{N}()

# Determin the output type
Base.BroadcastStyle(::Type{<:NDArray{T,N}}) where {T,N} = NDArrayStyle{N}()

Base.broadcastable(x::NDArray) = x

# Make it non-lazy
broadcasted(f, x::NDArray, args...)    = f(x, args...)
broadcasted(f, y, x::NDArray, args...) = f(y, x, args...)
broadcasted(f, x::NDArray{T,N}, y::NDArray{T,N}, args...) where {T,N} =
  f(x, y, args...)
