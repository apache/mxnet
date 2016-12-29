@defstruct NadamOptions <: AbstractOptimizerOptions (
  (lr                :: Real = 0.001, lr > 0),
  (beta1             :: Real = 0.99,  beta1 > 0 && beta1 < 1),
  (beta2             :: Real = 0.999,  beta2 > 0 && beta2 < 1),
  (epsilon           :: Real = 1e-8, epsilon > 0),
  (grad_clip         :: Real = 0, grad_clip >= 0),
  (weight_decay      :: Real = 0.00001, weight_decay >= 0),
  lr_scheduler       :: Any = nothing,
  momentum_scheduler :: Any = nothing
)

"""
    Nadam

Nesterov Adam optimizer: Adam RMSprop with Nesterov momentum,
see [1] and notes for further description.

    Nadam(; kwargs...)

# Attributes
* `lr::Real`: default `0.001`, learning rate.
* `beta1::Real`: default `0.99`.
* `beta2::Real`: default `0.999`.
* `epsilon::Real`: default `1e-8`, small value added for
  numerical stability
* `grad_clip::Real`: default `0`, if positive, will clip the gradient
  into the range `[-grad_clip, grad_clip]`.
* `weight_decay::Real`: default `0.00001`, weight decay is equivalent
  to adding a global l2 regularizer for all the parameters.
* `lr_scheduler::AbstractLearningRateScheduler`: default `nothing`, a
  dynamic learning rate scheduler. If set, will overwrite the `lr`
  parameter.
* `momentum_scheduler::AbstractMomentumScheduler` default
  `NadamScheduler` of the form
  ``\mu_t = beta1 * (1 - 0.5 * 0.96^{t * 0.004})``

# Notes
Default parameters follow those provided in the paper.
It is recommended to leave the parameters of this optimizer
at their default values.

# References
* [1]: Incorporating Nesterov Momentum into Adam.
  [http://cs229.stanford.edu/proj2015/054_report.pdf]
  (http://cs229.stanford.edu/proj2015/054_report.pdf)
* [2]: On the importance of initialization and momentum in deep learning
  [http://www.cs.toronto.edu/~fritz/absps/momentum.pdf]
  (http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
"""
type Nadam <: AbstractOptimizer
  opts  :: NadamOptions
  state :: OptimizationState

  function Nadam(; kwargs...)
    opts = NadamOptions(; kwargs...)
    opts.lr_scheduler = get_lr_scheduler(opts.lr_scheduler, opts.lr)
    opts.momentum_scheduler = get_momentum_scheduler(opts.momentum_scheduler,
      Momentum.NadamScheduler(mu0=opts.beta1))

    new(opts)
  end
end

type NadamState
  mt         :: NDArray
  nt         :: NDArray
  momentum   :: Float64
  beta2Power :: Float64
end

function create_state(self :: Nadam, index :: Int, weight :: NDArray)
  return NadamState( zeros(size(weight), context(weight)),
                     zeros(size(weight), context(weight)),
                     1.0,
                     self.opts.beta2 )
end

function update(self :: Nadam, index :: Int, weight :: NDArray,
                grad :: NDArray, state :: NadamState)
  lr = get_learning_rate(self.opts.lr_scheduler, self.state)
  grad = normalized_gradient(self.opts, self.state, weight, grad)

  mu_t, mu_t1 =
    get_momentum(self.opts.momentum_scheduler, self.state)
  state.momentum *= mu_t
  momentum_next = state.momentum * mu_t1

  grad_prime = grad / (1.0 - state.momentum)
  @inplace state.mt .*= self.opts.beta1
  @inplace state.mt .+= (1.0 - self.opts.beta1) * grad
  mt = state.mt / (1.0 - momentum_next)

  @inplace state.nt .*= self.opts.beta2
  @inplace state.nt .+= (1.0 - self.opts.beta2) * grad .* grad
  nt = state.nt / (1.0 - state.beta2Power)
  state.beta2Power *= self.opts.beta2

  mt_prime = (1.0 - mu_t) * grad_prime + mu_t1 * mt
  @inplace weight .+= -lr * mt_prime ./ (sqrt(nt) + self.opts.epsilon)
end
