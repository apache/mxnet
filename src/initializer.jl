abstract AbstractInitializer

function call(self :: AbstractInitializer, name :: Symbol, array :: NDArray)
  name = string(name)
  if endswith(name, "bias")
    _init_bias(self, name, array)
  elseif endswith(name, "gamma")
    _init_gamma(self, name, array)
  elseif endswith(name, "beta")
    _init_beta(self, name, array)
  elseif endswith(name, "weight")
    _init_weight(self, name, array)
  elseif endswith(name, "moving_mean")
    _init_zero(self, name, array)
  elseif endswith(name, "moving_var")
    _init_zero(self, name, array)
  else
    _init_default(self, name, array)
  end
end

function _init_bias(self :: AbstractInitializer, name :: Symbol, array :: NDArray)
  array[:] = 0
end
function _init_gamma(self :: AbstractInitializer, name :: Symbol, array :: NDArray)
  array[:] = 1
end
function _init_beta(self :: AbstractInitializer, name :: Symbol, array :: NDArray)
  array[:] = 0
end
function _init_zero(self :: AbstractInitializer, name :: Symbol, array :: NDArray)
  array[:] = 0
end

immutable UniformInitializer <: AbstractInitializer
  scale :: AbstractFloat
end
UniformInitializer() = UniformInitializer(0.07)

function _init_weight(self :: UniformInitializer, name :: Symbol, array :: NDArray)
  rand!(-self.scale, self.scale, array)
end

immutable NormalInitializer <: AbstractInitializer
  μ :: AbstractFloat
  σ :: AbstractFloat
end
NormalInitializer(; mu=0, sigma=0.01) = NormalInitializer(mu, sigma)

function _init_weight(self :: NormalInitializer, name :: Symbol, array :: NDArray)
  randn!(self.μ, self.σ, array)
end

immutable XaiverInitializer <: AbstractInitializer
end
function _init_weight(self :: NormalInitializer, name :: Symbol, array :: NDArray)
  dims    = size(array)
  fan_in  = prod(dims[2:end])
  fan_out = dims[1]
  scale   = sqrt(3 / (fan_in + fan_out))
  rand!(-scale, scale, array)
end
