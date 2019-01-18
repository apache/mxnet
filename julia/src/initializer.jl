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

"""
    AbstractInitializer

The abstract base class for all initializers.

To define a new initializer, it is
enough to derive a new type, and implement one or more of the following methods:

    _init_weight(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
    _init_bias(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
    _init_gamma(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
    _init_beta(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)

Or, if full behavior customization is needed, override the following function

    init(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
"""
abstract type AbstractInitializer end

function init(self :: T, name :: Base.Symbol, array :: NDArray) where T<:AbstractInitializer
  strname = string(name)
  if startswith(strname,"upsampling")
    _init_bilinear(self,name, array)
  elseif startswith(strname,"stn_loc") && endswith(strname,"weight")
    _init_zero(self,name, array)
  elseif startswith(strname,"stn_loc") && endswith(strname,"bias")
    _init_loc_bias(self,name, array)
  elseif endswith(strname, "bias")
    _init_bias(self, name, array)
  elseif endswith(strname, "gamma")
    _init_gamma(self, name, array)
  elseif endswith(strname, "beta")
    _init_beta(self, name, array)
  elseif endswith(strname, "weight")
    _init_weight(self, name, array)
  elseif endswith(strname, "moving_mean")
    _init_zero(self, name, array)
  elseif endswith(strname, "moving_var")
    _init_zero(self, name, array)
  else
    _init_default(self, name, array)
  end
end

function _init_loc_bias(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
 assert(size(array) == (6,))
 array[:]= [1.0, 0, 0, 0, 1.0, 0]
end

function _init_bilinear(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
  @assert ndims(array) == 4

  W, H, C, N = size(array) # Inverse of NCHW layout
  filter = Base.zeros(eltype(array), W, H)

  @assert H == W

  f = ceil(Int, W / 2) # factor
  c = (2 * f - 1 - f % 2) / (2 * f) # center
  for x in 0:(W-1)
    for y in 0:(H-1)
      filter[x+1, y+1] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
    end
  end

  @nd_as_jl rw=array begin
    for i in 1:N
      for j in 1:C
        array[:,:, j, i] = filter
      end
    end
  end
end

function _init_bias(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
  array[:] = 0
end
function _init_gamma(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
  array[:] = 1
end
function _init_beta(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
  array[:] = 0
end
function _init_zero(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
  array[:] = 0
end

function _init_default(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
  error("Do not know how to init $name")
end

"""
    UniformInitializer

Initialize weights according to a uniform distribution within the provided scale.
"""
struct UniformInitializer <: AbstractInitializer
  scale :: AbstractFloat
end
"""
    UniformInitializer(scale=0.07)

Construct a `UniformInitializer` with the specified scale.
"""
UniformInitializer() = UniformInitializer(0.07)

_init_weight(i::UniformInitializer, name::Symbol, x::NDArray) =
  rand!(x, low = -i.scale, high = i.scale)

"""
    NormalInitializer

Initialize weights according to a univariate Gaussian distribution.
"""
struct NormalInitializer <: AbstractInitializer
  μ :: AbstractFloat
  σ :: AbstractFloat
end
"""
    NormalInitializer(; mu=0, sigma=0.01)

Construct a `NormalInitializer` with mean `mu` and variance `sigma`.
"""
NormalInitializer(; mu=0, sigma=0.01) = NormalInitializer(mu, sigma)

_init_weight(i::NormalInitializer, name::Symbol, x::NDArray) =
  randn!(x, μ = i.μ, σ = i.σ)

@enum XavierDistribution xv_uniform xv_normal
@enum XavierRegularization xv_avg xv_in xv_out


"""
    XavierInitializer

The initializer documented in the paper [Bengio and Glorot 2010]: *Understanding
the difficulty of training deep feedforward neuralnetworks*.

There are several different version of the XavierInitializer used in the wild.
The general idea is that the variance of the initialization distribution is controlled
by the dimensionality of the input and output. As a distribution one can either choose
a normal distribution with μ = 0 and σ² or a uniform distribution from -σ to σ.

Several different ways of calculating the variance are given in the literature or are
used by various libraries.

* [Bengio and Glorot 2010]: `mx.XavierInitializer(distribution = mx.xv_uniform, regularization = mx.xv_avg, magnitude = 1)`
* [K. He, X. Zhang, S. Ren, and J. Sun 2015]: `mx.XavierInitializer(distribution = mx.xv_gaussian, regularization = mx.xv_in, magnitude = 2)`
* caffe_avg: `mx.XavierInitializer(distribution = mx.xv_uniform, regularization = mx.xv_avg, magnitude = 3)`
"""
struct XavierInitializer <: AbstractInitializer
  distribution :: XavierDistribution
  regularization :: XavierRegularization
  magnitude :: Float64
end

XavierInitializer(; distribution = xv_uniform, regularization = xv_avg, magnitude = 3.0) =
  XavierInitializer(distribution, regularization, magnitude)

function _init_weight(self :: XavierInitializer, name :: Base.Symbol, array :: NDArray)
  dims    = size(array)
  fan_in  = prod(dims[2:end])
  fan_out = dims[1]

  if self.regularization == xv_avg
    factor = (fan_in + fan_out) / 2
  elseif self.regularization == xv_in
    factor = fan_in
  elseif self.regularization == xv_out
    factor = fan_out
  end

  σ = √(self.magnitude / factor)

  if self.distribution == xv_uniform
    rand!(array, low = -σ, high = σ)
  elseif self.distribution == xv_normal
    randn!(array; μ = 0.0, σ = σ)
  end
end
