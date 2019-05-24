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

# trigonometric functions, remap to keep consistent API with Base
@_remap broadcasted(::typeof(sin),  x::NDArray) sin(x)
@_remap broadcasted(::typeof(cos),  x::NDArray) cos(x)
@_remap broadcasted(::typeof(tan),  x::NDArray) tan(x)
@_remap broadcasted(::typeof(asin), x::NDArray) arcsin(x)
@_remap broadcasted(::typeof(acos), x::NDArray) arccos(x)
@_remap broadcasted(::typeof(atan), x::NDArray) arctan(x)

# hyperbolic functions, remap to keep consistent API with Base
@_remap broadcasted(::typeof(sinh),  x::NDArray) sinh(x)
@_remap broadcasted(::typeof(cosh),  x::NDArray) cosh(x)
@_remap broadcasted(::typeof(tanh),  x::NDArray) tanh(x)
@_remap broadcasted(::typeof(asinh), x::NDArray) arcsinh(x)
@_remap broadcasted(::typeof(acosh), x::NDArray) arccosh(x)
@_remap broadcasted(::typeof(atanh), x::NDArray) arctanh(x)
