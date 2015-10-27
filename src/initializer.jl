#=doc
Built-in Initializers
=====================
=#

#=doc
.. class:: AbstractInitializer

   The abstract base class for all initializers.
=#
abstract AbstractInitializer

function call(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
  strname = string(name)
  if endswith(strname, "bias")
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

#=doc
.. class:: UniformInitializer

   Initialize weights according to a uniform distribution within the provided scale.
=#
immutable UniformInitializer <: AbstractInitializer
  scale :: AbstractFloat
end
UniformInitializer() = UniformInitializer(0.07)

function _init_weight(self :: UniformInitializer, name :: Base.Symbol, array :: NDArray)
  rand!(-self.scale, self.scale, array)
end

#=doc
.. class:: NormalInitializer

   Initialize weights according to a univariate Gaussian distribution.
=#
immutable NormalInitializer <: AbstractInitializer
  μ :: AbstractFloat
  σ :: AbstractFloat
end
NormalInitializer(; mu=0, sigma=0.01) = NormalInitializer(mu, sigma)

function _init_weight(self :: NormalInitializer, name :: Base.Symbol, array :: NDArray)
  randn!(self.μ, self.σ, array)
end

immutable XaiverInitializer <: AbstractInitializer
end

function _init_weight(self :: NormalInitializer, name :: Base.Symbol, array :: NDArray)
  dims    = size(array)
  fan_in  = prod(dims[2:end])
  fan_out = dims[1]
  scale   = sqrt(3 / (fan_in + fan_out))
  rand!(-scale, scale, array)
end
