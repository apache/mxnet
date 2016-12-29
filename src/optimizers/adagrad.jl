@defstruct AdaGradOptions <: AbstractOptimizerOptions (
  (lr           :: Real = 0.1, lr > 0),
  (epsilon      :: Real = 1e-6, epsilon > 0),
  (grad_clip    :: Real = 0, grad_clip >= 0),
  (weight_decay :: Real = 0.00001, weight_decay >= 0),
  lr_scheduler  :: Any  = nothing
)

"""
    AdaGrad

Scale learning rates by dividing with the square root of accumulated
squared gradients. See [1] for further description.

    AdaGrad(; kwargs...)

# Attributes
* `lr::Real`: default `0.1`, the learning rate controlling the
  size of update steps
* `epsilon::Real`: default `1e-6`, small value added for
  numerical stability
* `grad_clip::Real`: default `0`, if positive, will clip the gradient
  into the range `[-grad_clip, grad_clip]`.
* `weight_decay::Real`: default `0.00001`, weight decay is equivalent
  to adding a global l2 regularizer for all the parameters.

# Notes
Using step size lr AdaGrad calculates the learning rate for feature i at
time step t as:
``η_{t,i} = \frac{lr}{\sqrt{\sum^t_{t^\prime} g^2_{t^\prime,i} + ϵ}} g_{t,i}``
as such the learning rate is monotonically decreasing.
Epsilon is not included in the typical formula, see [2].

# References
* [1]: Duchi, J., Hazan, E., & Singer, Y. (2011):
  Adaptive subgradient methods for online learning and
  stochastic optimization. JMLR, 12:2121-2159.
* [2]: Chris Dyer: Notes on AdaGrad.
  [http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf]
  (http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf)
"""

type AdaGrad <: AbstractOptimizer
  opts  :: AdaGradOptions
  state :: OptimizationState

  function AdaGrad(; kwargs...)
    opts = AdaGradOptions(;kwargs...)
    opts.lr_scheduler = get_lr_scheduler(opts.lr_scheduler, opts.lr)

    new(opts)
  end
end

function create_state(self :: AdaGrad, index :: Int, weight :: NDArray)
  return zeros(size(weight), context(weight))
end

function update(self :: AdaGrad, index :: Int, weight :: NDArray,
                grad :: NDArray, state :: NDArray)
  lr = get_learning_rate(self.opts.lr_scheduler, self.state)
  grad = normalized_gradient(self.opts, self.state, weight, grad)

  @inplace state .+= grad .* grad
  @inplace weight .+= -lr * grad ./ (sqrt(state + self.opts.epsilon))
end
