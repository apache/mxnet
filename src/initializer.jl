#=doc
Initializers
============
Interface
---------
=#

#=doc
.. class:: AbstractInitializer

   The abstract base class for all initializers.

To define a new initializer, it is
enough to derive a new type, and implement one or more of the following methods:

.. function:: _init_weight(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
.. function:: _init_bias(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
.. function:: _init_gamma(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
.. function:: _init_beta(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)

Or, if full behavior customization is needed, override the following function

.. function:: call(self :: AbstractInitializer, name :: Base.Symbol, array :: NDArray)
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
Built-in initializers
---------------------
=#
#=doc
.. class:: UniformInitializer

   Initialize weights according to a uniform distribution within the provided scale.
=#
immutable UniformInitializer <: AbstractInitializer
  scale :: AbstractFloat
end
#=doc
.. function UniformInitializer(scale=0.07)

   Construct a :class:`UniformInitializer` with the specified scale.
=#
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
#=doc
.. function:: NormalIninitializer(; mu=0, sigma=0.01)

   Construct a :class:`NormalInitializer` with mean ``mu`` and variance ``sigma``.
=#
NormalInitializer(; mu=0, sigma=0.01) = NormalInitializer(mu, sigma)

function _init_weight(self :: NormalInitializer, name :: Base.Symbol, array :: NDArray)
  randn!(self.μ, self.σ, array)
end

#=doc
.. class:: XavierInitializer

   The initializer documented in the paper [Bengio and Glorot 2010]: *Understanding
   the difficulty of training deep feedforward neuralnetworks*.

   There are several different version of the XavierInitializer used in the wild.
   The general idea is that the variance of the initialization distribution is controlled
   by the dimensionality of the input and output. As a distribution one can either choose
   a normal distribution with μ = 0 and σ² or a uniform distribution from -σ to σ.

   Several different ways of calculating the variance are given in the literature or are
   used by various libraries.

   - original [Bengio and Glorot 2010]: σ² = 2 / (in + out)
   - msra [K. He, X. Zhang, S. Ren, and J. Sun 2015]: σ² = 2 / in
   - caffe_avg: 6 / (in + out)
   - caffe_in: 3 / in
   - caffe_out: 3 / out
   - mxnet: 3 / (in + out)

   Distribution and variant can be chosen by enums (prefixed by ``xv_``).
   As an example take ``mx.XavierInitializer(distribution = mx.xv_normal, variant = mx.xv_mxnet)``,
   which is currently the default.
=#

@enum XavierDistribution xv_uniform xv_normal
@enum XavierVariant xv_original xv_mrsa xv_caffe_avg xv_caffe_in zv_caffe_out xv_mxnet

immutable XavierInitializer <: AbstractInitializer
  distribution :: XavierDistribution
  variant :: XavierVariant
end
XavierInitializer(; distribution = xv_uniform, variant = xv_mxnet) = XavierInitializer(distribution, variant)

function _init_weight(self :: XavierInitializer, name :: Base.Symbol, array :: NDArray)
  dims    = size(array)
  fan_in  = prod(dims[2:end])
  fan_out = dims[1]

  if self.distribution == xv_uniform
    func(σ, data) = rand!(-σ, σ, data)
  elseif self.distribution == xv_normal
    func(σ, data) = randn!(0.0, σ, data)
  end

  if self.variant == xv_caffe_avg
    var = 6 / (fan_in + fan_out)
  elseif self.variant == xv_caffe_in
    var = 3 / fan_in
  elseif self.variant == xv_caffe_out
    var = 3 / fan_out
  elseif self.variant == xv_mrsa
    var = 2 / fan_in
  elseif self.variant == xv_original
    var = 2 / (fan_in + fan_out)
  elseif self.variant == xv_mxnet
    var = 3 / (fan_in + fan_out)
  end

  func(√var, array)
end
