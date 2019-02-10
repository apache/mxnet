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

broadcasted(::typeof(==), x::NDArray{T}, y::NDArray{T}) where {T} =
  _broadcast_equal(x, y)

broadcasted(::typeof(!=), x::NDArray{T}, y::NDArray{T}) where {T} =
  _broadcast_not_equal(x, y)

broadcasted(::typeof(>), x::NDArray{T}, y::NDArray{T}) where {T} =
  _broadcast_greater(x, y)

broadcasted(::typeof(>=), x::NDArray{T}, y::NDArray{T}) where {T} =
  _broadcast_greater_equal(x, y)

broadcasted(::typeof(<), x::NDArray{T}, y::NDArray{T}) where {T} =
  _broadcast_lesser(x, y)

broadcasted(::typeof(<=), x::NDArray{T}, y::NDArray{T}) where {T} =
  _broadcast_lesser_equal(x, y)

################################################################################
# remapping to solving type unstablility
################################################################################

@_remap _broadcast_equal(x::NDArray, y::NDArray)  broadcast_equal(x, y)
@_remap _broadcast_equal!(x::NDArray, y::NDArray) broadcast_equal(x, y)

@_remap _broadcast_not_equal(x::NDArray, y::NDArray)  broadcast_not_equal(x, y)
@_remap _broadcast_not_equal!(x::NDArray, y::NDArray) broadcast_not_equal(x, y)

@_remap _broadcast_greater(x::NDArray, y::NDArray)  broadcast_greater(x, y)
@_remap _broadcast_greater!(x::NDArray, y::NDArray) broadcast_greater(x, y)

@_remap _broadcast_greater_equal(x::NDArray, y::NDArray)  broadcast_greater_equal(x, y)
@_remap _broadcast_greater_equal!(x::NDArray, y::NDArray) broadcast_greater_equal(x, y)

@_remap _broadcast_lesser(x::NDArray, y::NDArray)  broadcast_lesser(x, y)
@_remap _broadcast_lesser!(x::NDArray, y::NDArray) broadcast_lesser(x, y)

@_remap _broadcast_lesser_equal(x::NDArray, y::NDArray)  broadcast_lesser_equal(x, y)
@_remap _broadcast_lesser_equal!(x::NDArray, y::NDArray) broadcast_lesser_equal(x, y)
