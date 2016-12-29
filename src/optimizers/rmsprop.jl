@defstruct RMSPropOptions <: AbstractOptimizerOptions (
  (lr           :: Real = 0.001, lr > 0),
  (rho          :: Real = 0.9, rho > 0 && rho < 1),
  (epsilon      :: Real = 1e-6, epsilon > 0),
  (grad_clip    :: Real = 0, grad_clip >= 0),
  (weight_decay :: Real = 0.00001, weight_decay >= 0),
  lr_scheduler  :: Any  = nothing
)

"""
    RMSProp

Scale learning rates by dividing with the moving average of the root mean
squared (RMS) gradients. See [1] for further description.

    RMSProp(; kwargs...)

# Attributes
* `lr::Real`: default `0.1`, the learning rate controlling the
  size of update steps
* `rho::Real`: default `0.9`, gradient moving average decay factor
* `epsilon::Real`: default `1e-6`, small value added for
  numerical stability
* `grad_clip::Real`: default `0`, if positive, will clip the gradient
  into the range `[-grad_clip, grad_clip]`.
* `weight_decay::Real`: default `0.00001`, weight decay is equivalent
  to adding a global l2 regularizer for all the parameters.

# Notes
`rho` should be between 0 and 1. A value of `rho` close to 1 will decay the
moving average slowly and a value close to 0 will decay the moving average
fast.

Using the step size ``lr`` and a decay factor ``\rho`` the
learning rate ``\eta_t`` is calculated as:
``r_t &= ρ r_{t-1} + (1 - ρ)*g^2 \\
  η_t &= \frac{lr}{\sqrt{r_t + ϵ}}``

# References
* [1]: Tieleman, T. and Hinton, G. (2012):
  Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
  Coursera. [http://www.youtube.com/watch?v=O3sxAc4hxZU]
  (http://www.youtube.com/watch?v=O3sxAc4hxZU) (formula @5:20)
"""

type RMSProp <: AbstractOptimizer
  opts  :: RMSPropOptions
  state :: OptimizationState

  function RMSProp(; kwargs...)
    opts = RMSPropOptions(;kwargs...)
    opts.lr_scheduler = get_lr_scheduler(opts.lr_scheduler, opts.lr)

    new(opts)
  end
end

function create_state(self :: RMSProp, index :: Int, weight :: NDArray)
  return zeros(size(weight), context(weight))
end

function update(self :: RMSProp, index :: Int, weight :: NDArray,
                grad :: NDArray, state :: NDArray)
  lr = get_learning_rate(self.opts.lr_scheduler, self.state)
  grad = normalized_gradient(self.opts, self.state, weight, grad)

  @inplace state .*= self.opts.rho
  @inplace state .+= (1 - self.opts.rho) * grad .* grad

  @inplace weight .+= -lr * grad ./ (sqrt(state + self.opts.epsilon))
end
