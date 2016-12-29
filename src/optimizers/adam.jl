@defstruct ADAMOptions <: AbstractOptimizerOptions (
  (lr           :: Real = 0.001, lr > 0),
  (grad_clip    :: Real = 0, grad_clip >= 0),
  (weight_decay :: Real = 0.00001, weight_decay >= 0),
  (beta1        :: Real = 0.9,  beta1 > 0),
  (beta2        :: Real = 0.999,  beta2 > 0),
  (epsilon      :: Real = 1e-8, epsilon > 0),
  lr_scheduler  :: Any  = nothing
)

"""
     ADAM

The solver described in Diederik Kingma, Jimmy Ba: *Adam: A Method for
Stochastic Optimization*. arXiv:1412.6980 [cs.LG].

    ADAM(; kwargs...)

* `lr::Real`: default `0.001`, learning rate.
* `lr_scheduler::AbstractLearningRateScheduler`: default `nothing`, a
       dynamic learning rate scheduler. If set, will overwrite the `lr`
       parameter.
* `beta1::Real`: default `0.9`.
* `beta2::Real`: default `0.999`.
* `epsilon::Real`: default `1e-8`.
* `grad_clip::Real`: default `0`, if positive, will clip the gradient
       into the range `[-grad_clip, grad_clip]`.
* `weight_decay::Real`: default `0.00001`, weight decay is equivalent
       to adding a global l2 regularizer for all the parameters.
"""
type ADAM <: AbstractOptimizer
  opts  :: ADAMOptions
  state :: OptimizationState

  function ADAM(; kwargs...)
    opts = ADAMOptions(;kwargs...)
    opts.lr_scheduler = get_lr_scheduler(opts.lr_scheduler, opts.lr)

    new(opts)
  end
end

type ADAMState
  current_lr :: Float64  # current learning rate
  mt         :: NDArray
  vt         :: NDArray
  beta1Power :: Float64
  beta2Power :: Float64
end

function create_state(self :: ADAM, index :: Int, weight :: NDArray)
  return ADAMState( get_learning_rate(self.opts.lr_scheduler, self.state),
                    zeros(size(weight), context(weight)),
                    zeros(size(weight), context(weight)),
                    self.opts.beta1,
                    self.opts.beta2 )
end

function update(self :: ADAM, index :: Int, weight :: NDArray, grad :: NDArray, state :: ADAMState)
  lr = state.current_lr
  grad = normalized_gradient(self.opts, self.state, weight, grad)

  state.mt = self.opts.beta1 * state.mt + (1 - self.opts.beta1) * grad
  state.vt = self.opts.beta2 * state.vt + (1 - self.opts.beta2) * (grad .* grad)

  at = sqrt(1.0 - state.beta2Power)/(1.0 - state.beta1Power)

  state.beta1Power *= self.opts.beta1
  state.beta2Power *= self.opts.beta2

  @inplace weight .+= -lr * at * state.mt ./
    (sqrt(state.vt) + self.opts.epsilon)
end
