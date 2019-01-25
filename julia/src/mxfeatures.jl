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

module MXFeatures

using ..mx

export Feature
export has_feature, features_enabled

@enum Feature begin
  CUDA
  CUDNN
  NCCL
  CUDA_RTC
  TENSORRT
  CPU_SSE
  CPU_SSE2
  CPU_SSE3
  CPU_SSE4_1
  CPU_SSE4_2
  CPU_SSE4A
  CPU_AVX
  CPU_AVX2
  OPENMP
  SSE
  F16C
  JEMALLOC
  BLAS_OPEN
  BLAS_ATLAS
  BLAS_MKL
  BLAS_APPLE
  LAPACK
  MKLDNN
  OPENCV
  CAFFE
  PROFILER
  DIST_KVSTORE
  CXX14
  SIGNAL_HANDLER
  DEBUG
end

"""
    features_enabled()
    features_enabled(Symbol)
    features_enabled(String; sep = ", ")

Returns a list of enabled features in the back-end


## Examples

```julia-repl
julia> mx.features_enabled()
8-element Array{MXNet.mx.MXFeatures.Feature,1}:
 CPU_SSE::Feature = 5
 CPU_SSE2::Feature = 6
 CPU_SSE3::Feature = 7
 CPU_SSE4_1::Feature = 8
 CPU_SSE4_2::Feature = 9
 CPU_AVX::Feature = 11
 F16C::Feature = 15
 LAPACK::Feature = 21

julia> mx.features_enabled(String)
"CPU_SSE, CPU_SSE2, CPU_SSE3, CPU_SSE4_1, CPU_SSE4_2, CPU_AVX, F16C, LAPACK"

julia> mx.features_enabled(Symbol)
8-element Array{Symbol,1}:
 :CPU_SSE
 :CPU_SSE2
 :CPU_SSE3
 :CPU_SSE4_1
 :CPU_SSE4_2
 :CPU_AVX
 :F16C
 :LAPACK
```
"""
features_enabled() = filter(f -> has_feature(f), collect(instances(Feature)))
features_enabled(::Type{String}; sep = ", ") = join(features_enabled(Symbol), sep)
features_enabled(::Type{Symbol}) = Symbol.(features_enabled())


"""
    hasfeature(feature::Feature) -> Bool

Check the library for compile-time feature at runtime
"""
function has_feature(x::Feature)
  y = Ref{Bool}(0)
  @mx.mxcall(:MXHasFeature, (mx.MX_uint, Ref{Bool},), mx.MX_uint(x), y)
  y[]
end

end  # module MXFeatures

using .MXFeatures
