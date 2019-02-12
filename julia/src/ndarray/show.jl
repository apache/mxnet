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

function Base.show(io::IO, x::NDArray)
  print(io, "NDArray(")
  Base.show(io, try_get_shared(x, sync = :read))
  print(io, ")")
end

# for REPL
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, x::NDArray{T,N}) where {T,N}
  type_ = split(string(typeof(x)), '.', limit=2)[end]
  n = length(x)
  size_ = N == 1 ? "$n-element" : join(size(x), "×")
  print(io, "$size_ $type_ @ $(context(x))", (n == 0) ? "" : ":\n")
  Base.print_array(io, try_get_shared(x, sync = :read))
end
