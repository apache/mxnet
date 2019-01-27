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

# activation functions

@doc doc"""
    σ.(x::NDArray)
    sigmoid.(x::NDArray)

Computes sigmoid of x element-wise.

```math
σ(x) = \frac{1}{(1 + exp(-x))}
```

The storage type of `sigmoid` output is always dense.
"""
function σ end
const sigmoid = σ
_nddoc[:σ] = false
@_remap broadcasted(::typeof(σ), x::NDArray) sigmoid(x)

@doc doc"""
    relu.(x::NDArray)

Computes rectified linear.

```math
\max(x, 0)
```
"""
function relu end
_nddoc[:relu] = false
@_remap broadcasted(::typeof(relu), x::NDArray) relu(x)

@doc doc"""
    softmax.(x::NDArray, [dim = ndims(x)])

Applies the softmax function.

The resulting array contains elements in the range `(0, 1)`
and the elements along the given axis sum up to 1.

```math
softmax(\mathbf{z})_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}
```
"""
function softmax end
_nddoc[:softmax] = false
@_remap broadcasted(::typeof(softmax), x::NDArray)           softmax(x; axis = -ndims(x))
@_remap broadcasted(::typeof(softmax), x::NDArray, dim::Int) softmax(x; axis = -dim)

"""
    log_softmax.(x::NDArray, [dim = ndims(x)])

Computes the log softmax of the input.
This is equivalent to computing softmax followed by log.

julia> x
2×3 mx.NDArray{Float64,2} @ CPU0:
 1.0  2.0  0.1
 0.1  2.0  1.0

julia> mx.log_softmax.(x)
2×3 mx.NDArray{Float64,2} @ CPU0:
 -1.41703  -0.41703  -2.31703
 -2.31703  -0.41703  -1.41703
"""
function log_softmax end
_nddoc[:log_softmax] = false
@_remap broadcasted(::typeof(log_softmax), x::NDArray)           log_softmax(x; axis = -ndims(x))
@_remap broadcasted(::typeof(log_softmax), x::NDArray, dim::Int) log_softmax(x; axis = -dim)

