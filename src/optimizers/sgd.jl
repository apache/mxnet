@defstruct SGDOptions Any (
  (lr           :: Real = 0.01, lr > 0),
  (momentum     :: Real = 0.0, momentum >= 0),
  (weight_decay :: Real = 0.0001, weight_decay >= 0),
  (grad_scale   :: Real = 1.0, grad_scale >= 0),
  (grad_clip    :: Real = 0, grad_clip >= 0),
  lr_scheduler  :: Any  = nothing,
  mom_scheduler :: Any  = nothing
)


type SGD <: AbstractOptimizer
  iter       :: Int
  batch_size :: Int
  opts       :: SGDOptions

  function SGD(; kwargs...)
    opts = SGDOptions(;kwargs...)
    if !isa(opts.lr_scheduler, AbstractLearningRateScheduler)
      opts.lr_scheduler = FixedLearningRateScheduler(opts.lr)
    end
    if !isa(opts.mom_scheduler, AbstractMomentumScheduler)
      opts.mom_scheduler = opts.momentum > 0 ?
              FixedMomentumScheduler(opts.momentum) :
              NullMomentumScheduler()
    end

    new(0, 0, opts)
  end
end

function create_state(self :: SGD, index :: Int, weight :: NDArray)
  if isa(self.opts.mom_scheduler, NullMomentumScheduler)
    return nothing
  else
    return zeros(size(weight), context(weight))
  end
end

function update(self :: SGD, index :: Int, weight :: NDArray, grad :: NDArray, state :: Union{Void, NDArray})
  lr = get_learning_rate(self.opts.lr_scheduler, self.iter)
  grad_scale = self.opts.grad_scale / self.batch_size

  grad = grad_scale * grad
  if self.opts.grad_clip > 0
    grad = clip(grad, -self.opts.grad_clip, self.opts.grad_clip)
  end

  if isa(state, Void)
    @inplace weight += -lr * (grad + self.opts.weight_decay * weight)
  else
    mom = state :: NDArray
    coef = get_momentum(self.opts.mom_scheduler, self.iter)
    @inplace mom    .*= coef
    @inplace mom    .+= -lr * (grad + self.opts.weight_decay * weight)
    @inplace weight .+= mom
  end
end
