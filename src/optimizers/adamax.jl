@defstruct AdaMaxOptions <: AbstractOptimizerOptions (
  (lr           :: Real = 0.002,    lr > 0),
  (beta1        :: Real = 0.9,      beta1 > 0 && beta1 < 1),
  (beta2        :: Real = 0.999,    beta2 > 0 && beta2 < 1),
  (epsilon      :: Real = 1e-8,     epsilon > 0),
  (grad_clip    :: Real = 0,        grad_clip >= 0),
  (weight_decay :: Real = 0.00001,  weight_decay >= 0),
  lr_scheduler  :: Any  = nothing
)

"""
    AdaMax

This is a variant of of the Adam algorithm based on the infinity norm.
See [1] for further description.

    AdaMax(; kwargs...)

# Attributes
* `lr::Real`: default `0.002`, the learning rate controlling the
  size of update steps
* `beta1::Real`: default `0.9`, exponential decay rate
  for the first moment estimates
* `beta2::Real`: default `0.999`, exponential decay rate for the
  weighted infinity norm estimates
* `epsilon::Real`: default `1e-8`, small value added for
  numerical stability
* `grad_clip::Real`: default `0`, if positive, will clip the gradient
  into the range `[-grad_clip, grad_clip]`.
* `weight_decay::Real`: default `0.00001`, weight decay is equivalent
  to adding a global l2 regularizer for all the parameters.

# References
* [1]: Kingma, Diederik, and Jimmy Ba (2014):
  Adam: A Method for Stochastic Optimization.
  [http://arxiv.org/abs/1412.6980v8]
  (http://arxiv.org/abs/1412.6980v8).
"""

type AdaMax <: AbstractOptimizer
  opts  :: AdaMaxOptions
  state :: OptimizationState

  function AdaMax(; kwargs...)
    opts = AdaMaxOptions(; kwargs...)
    opts.lr_scheduler = get_lr_scheduler(opts.lr_scheduler, opts.lr)

    new(opts)
  end
end

type AdaMaxState
  mt         :: NDArray
  ut         :: NDArray
  beta1Power :: Float64
end

function create_state(self :: AdaMax, index :: Int, weight :: NDArray)
  return AdaMaxState( zeros(size(weight), context(weight)),
                      zeros(size(weight), context(weight)),
                      self.opts.beta1 )
end

function update(self :: AdaMax, index :: Int, weight :: NDArray,
                grad :: NDArray, state :: AdaMaxState)
  lr = get_learning_rate(self.opts.lr_scheduler, self.state)
  grad = normalized_gradient(self.opts, self.state, weight, grad)

  @inplace state.mt .*= self.opts.beta1
  @inplace state.mt .+= (1 - self.opts.beta1) * grad
  state.ut = _maximum(self.opts.beta2 * state.ut, abs(grad))

  @inplace weight .+= - lr / (1 - state.beta1Power) *
    state.mt ./ (state.ut + self.opts.epsilon)

  state.beta1Power *= self.opts.beta1
end
