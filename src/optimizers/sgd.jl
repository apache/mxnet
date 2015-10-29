@defstruct SGDOptions AbstractOptimizerOptions (
  (lr                :: Real = 0.01, lr > 0),
  (momentum          :: Real = 0.0, momentum >= 0),
  (grad_scale        :: Real = 1.0, grad_scale >= 0),
  (grad_clip         :: Real = 0, grad_clip >= 0),
  (weight_decay      :: Real = 0.0001, weight_decay >= 0),
  lr_scheduler       :: Any  = nothing,
  momentum_scheduler :: Any  = nothing
)


type SGD <: AbstractOptimizer
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

function update(self :: SGD, index :: Int, weight :: NDArray, grad :: NDArray, state :: Union{Void, NDArray})
  lr = get_learning_rate(self.opts.lr_scheduler, self.state)
  grad = normalized_gradient(self.opts, self.state, weight, grad)

  if isa(state, Void)
    # vanilla SGD, without momentum
    @inplace weight += -lr * grad
  else
    mom = state :: NDArray
    coef = get_momentum(self.opts.momentum_scheduler, self.state)
    @inplace mom    .*= coef
    @inplace mom    .+= -lr * grad
    @inplace weight .+= mom
  end
end
