# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# License); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# runtime detection of compile time features in the native library

module MXRuntime

using ..mx

export LibFeature
export libinfo_features, isenabled

# defined in include/mxnet/c_api.h
struct LibFeature
  name::Ptr{Cchar}
  index::UInt32
  enabled::Bool
end

Base.show(io::IO, x::LibFeature) =
  print(io, ifelse(x.enabled, "✔", "✖"), "  ", unsafe_string(x.name))

"""
    libinfo_features()

Check the library for compile-time features.
The list of features are maintained in libinfo.h and libinfo.cc
"""
function libinfo_features()
  ref = Ref{Ptr{LibFeature}}(C_NULL)
  s = Ref{Csize_t}(C_NULL)
  @mx.mxcall(:MXLibInfoFeatures, (Ref{Ptr{LibFeature}}, Ref{Csize_t}), ref, s)
  unsafe_wrap(Array, ref[], s[])
end

"""
    isenabled(x::Symbol)::Bool

Returns the given runtime feature is enabled or not.

```julia-repl
julia> mx.isenabled(:CUDA)
false

julia> mx.isenabled(:CPU_SSE)
true
```

See also `mx.libinfo_features`.
"""
isenabled(x::Symbol) =
  any(libinfo_features()) do i
    Symbol(unsafe_string(i.name)) == x && i.enabled
  end

end  # module MXRuntime

using .MXRuntime
