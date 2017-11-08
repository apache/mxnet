@defstruct SGDOptions <: AbstractOptimizerOptions (
  (lr                :: Real = 0.01, lr > 0),
  (momentum          :: Real = 0.0, momentum >= 0),
  (grad_clip         :: Real = 0, grad_clip >= 0),
  (weight_decay      :: Real = 0.0001, weight_decay >= 0),
  lr_scheduler       :: Any  = nothing,
  momentum_scheduler :: Any  = nothing
)

"""
    SGD

Stochastic gradient descent optimizer.

    SGD(; kwargs...)

# Arguments:
* `lr::Real`: default `0.01`, learning rate.
* `lr_scheduler::AbstractLearningRateScheduler`: default `nothing`, a
       dynamic learning rate scheduler. If set, will overwrite the `lr`
       parameter.
* `momentum::Real`: default `0.0`, the momentum.
* `momentum_scheduler::AbstractMomentumScheduler`: default `nothing`,
       a dynamic momentum scheduler. If set, will overwrite the `momentum`
       parameter.
* `grad_clip::Real`: default `0`, if positive, will clip the gradient
       into the bounded range `[-grad_clip, grad_clip]`.
* `weight_decay::Real`: default `0.0001`, weight decay is equivalent to
       adding a global l2 regularizer to the parameters.
"""
mutable struct SGD <: AbstractOptimizer
  opts  :: SGDOptions
  state :: OptimizationState

  function SGD(; kwargs...)
    opts = SGDOptions(;kwargs...)
    opts.lr_scheduler = get_lr_scheduler(opts.lr_scheduler, opts.lr)
    opts.momentum_scheduler = get_momentum_scheduler(opts.momentum_scheduler, opts.momentum)

    new(opts)
  end
end

function create_state(self :: SGD, index :: Int, weight :: NDArray)
  if isa(self.opts.momentum_scheduler, Momentum.Null)
    return nothing
  else
    return zeros(size(weight), context(weight))
  end
end

function update(self :: SGD, index :: Int, weight :: NDArray, grad :: NDArray, state :: Void)
  lr = get_learning_rate(self.opts.lr_scheduler, self.state)
  grad = normalized_gradient(self.opts, self.state, weight, grad)
  
  @inplace weight += -lr * grad
end

# update with momentum
function update(self :: SGD, index :: Int, weight :: NDArray, grad :: NDArray, state :: NDArray)
  lr = get_learning_rate(self.opts.lr_scheduler, self.state)
  grad = normalized_gradient(self.opts, self.state, weight, grad)

  mom = state :: NDArray
  coef = get_momentum(self.opts.momentum_scheduler, self.state)
  @inplace mom    .*= coef
  @inplace mom    .+= -lr * grad
  @inplace weight .+= mom
end
