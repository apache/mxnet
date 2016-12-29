@defstruct AdaDeltaOptions <: AbstractOptimizerOptions (
  (lr           :: Real = 1.0, lr > 0),
  (rho          :: Real = 0.95, rho > 0 && rho < 1),
  (epsilon      :: Real = 1e-6, epsilon > 0),
  (grad_clip    :: Real = 0, grad_clip >= 0),
  (weight_decay :: Real = 0.00001, weight_decay >= 0),
  lr_scheduler  :: Any  = nothing
)

"""
    AdaDelta

Scale learning rates by the ratio of accumulated gradients to accumulated
updates, see [1] and notes for further description.

    AdaDelta(; kwargs...)

# Attributes
* `lr::Real`: default `1.0`, the learning rate controlling the
  size of update steps
* `rho::Real`: default `0.9`, squared gradient moving average decay factor
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

`rho` = 0.95 and `epsilon` = 1e-6 are suggested in the paper and reported to
work for multiple datasets (MNIST, speech). In the paper, no learning rate is
considered (so `lr` = 1.0). Probably best to keep it at this value.

`epsilon` is important for the very first update (so the numerator does
not become 0).

Using the step size `lr` and a decay factor `rho` the learning rate is
calculated as:
``r_t &= \rho r_{t-1} + (1-\rho)*g^2\\
\eta_t &= \eta \frac{\sqrt{s_{t-1} + \epsilon}} {\sqrt{r_t + \epsilon}}\\
s_t &= \rho s_{t-1} + (1-\rho)*(\eta_t*g)^2``

# References
* [1]: Zeiler, M. D. (2012):
  ADADELTA: An Adaptive Learning Rate Method. arXiv Preprint arXiv:1212.5701.
"""

type AdaDelta <: AbstractOptimizer
  opts  :: AdaDeltaOptions
  state :: OptimizationState

  function AdaDelta(; kwargs...)
    opts = AdaDeltaOptions(;kwargs...)
    opts.lr_scheduler = get_lr_scheduler(opts.lr_scheduler, opts.lr)

    new(opts)
  end
end

type AdaDeltaState
  acc       :: NDArray
  delta_acc :: NDArray
end

function create_state(self :: AdaDelta, index :: Int, weight :: NDArray)
  return AdaDeltaState(zeros(size(weight), context(weight)),
                       zeros(size(weight), context(weight)))
end

function update(self :: AdaDelta, index :: Int, weight :: NDArray,
                grad :: NDArray, state :: AdaDeltaState)
  lr = get_learning_rate(self.opts.lr_scheduler, self.state)
  grad = normalized_gradient(self.opts, self.state, weight, grad)

  # Update state.acc as in RMSProp
  @inplace state.acc .*= self.opts.rho
  @inplace state.acc .+= (1 - self.opts.rho) * grad .* grad

  # Compute update using the "old" state.delta_acc
  update = grad .* sqrt(state.delta_acc + self.opts.epsilon) ./
    (sqrt(state.acc + self.opts.epsilon))
  @inplace weight .+= -lr * update

  # update state.delta_acc using update
  @inplace state.delta_acc .*= self.opts.rho
  @inplace state.delta_acc .+= (1 - self.opts.rho) * update .* update
end
