"""
    AbstractOptimizer

Base type for all optimizers.
"""
abstract AbstractOptimizer

"""
    AbstractLearningRateScheduler

Base type for all learning rate scheduler.
"""
abstract AbstractLearningRateScheduler

"""
    AbstractMomentumScheduler

Base type for all momentum scheduler.
"""
abstract AbstractMomentumScheduler



"""
    OptimizationState

# Attributes:
* `batch_size`: The size of the mini-batch used in stochastic training.
* `curr_epoch`:
  The current epoch count. Epoch 0 means no training yet, during the first
  pass through the data, the epoch will be 1; during the second pass, the
  epoch count will be 1, and so on.
* `curr_batch`:
  The current mini-batch count. The batch count is reset during every epoch.
  The batch count 0 means the beginning of each epoch, with no mini-batch
  seen yet. During the first mini-batch, the mini-batch count will be 1.
* `curr_iter`:
  The current iteration count. One iteration corresponds to one mini-batch,
  but unlike the mini-batch count, the iteration count does **not** reset
  in each epoch. So it track the *total* number of mini-batches seen so far.
"""
type OptimizationState
  batch_size :: Int
  curr_epoch :: Int
  curr_batch :: Int
  curr_iter  :: Int
end
OptimizationState(batch_size::Int) = OptimizationState(batch_size, 0, 0, 0)


"""
    get_learning_rate(scheduler, state)

# Arguments
* `scheduler::AbstractLearningRateScheduler`: a learning rate scheduler.
* `state::OptimizationState`: the current state about epoch, mini-batch and iteration count.

Returns the current learning rate.
"""
function get_learning_rate end

################################################################################
# The learning rate module
module LearningRate
import ..mx: AbstractLearningRateScheduler, OptimizationState, get_learning_rate

"""
    LearningRate.Fixed

Fixed learning rate scheduler always return the same learning rate.
"""
type Fixed <: AbstractLearningRateScheduler
  learning_rate :: Float64
end
get_learning_rate(self :: Fixed, state :: OptimizationState) = self.learning_rate

"""
    LearningRate.Exp

``\eta_t = \eta_0\gamma^t``. Here ``t`` is the epoch count, or the iteration
count if `decay_on_iteration` is set to true.
"""
type Exp <: AbstractLearningRateScheduler
  learning_rate :: Float64
  gamma         :: Float64
  on_iteration  :: Bool
end
function Exp(base_lr::Real; gamma::Real=0.9, decay_on_iteration::Bool=false)
  @assert(0 < gamma < 1)
  Exp(Float64(base_lr), Float64(gamma), decay_on_iteration)
end
get_learning_rate(self :: Exp, state :: OptimizationState) =
    self.learning_rate * self.gamma ^ (self.on_iteration ? state.curr_iter : state.curr_epoch)
"""
    LearningRate.Inv

``\eta_t = \eta_0 * (1 + \gamma * t)^(-power)``.
Here ``t`` is the epoch count, or the iteration count if `decay_on_iteration`
is set to true.
"""
type Inv <: AbstractLearningRateScheduler
  learning_rate :: Float64
  gamma         :: Float64
  power         :: Float64
  on_iteration  :: Bool
end
function Inv(base_lr :: Real; gamma::Real=0.9, power::Real=0.5, decay_on_iteration::Bool=false)
  @assert(0 < gamma < 1)
  @assert(0 <= power)
  Inv(Float64(base_lr), Float64(gamma), Float64(power), decay_on_iteration)
end
get_learning_rate(self :: Inv, state :: OptimizationState) =
  self.learning_rate * ( 1 + self.gamma * (self.on_iteration ? state.curr_iter : state.curr_epoch)) ^ (-self.power)
end# module LearningRate
################################################################################
function get_lr_scheduler(scheduler :: Any, lr :: Real)
  if isa(scheduler, AbstractLearningRateScheduler)
    return scheduler
  else
    return LearningRate.Fixed(lr)
  end
end


"""
    get_momentum(scheduler, state)

* `scheduler::AbstractMomentumScheduler`: the momentum scheduler.
* `state::OptimizationState`: the state about current epoch, mini-batch and iteration count.

Returns the current momentum.
"""
function get_momentum
end


################################################################################
# The Momentum module
module Momentum
import ..mx: AbstractMomentumScheduler, OptimizationState, get_momentum

"""
    Momentum.Null

The null momentum scheduler always returns 0 for momentum. It is also used to
explicitly indicate momentum should not be used.
"""
type Null <: AbstractMomentumScheduler
end
get_momentum(self :: Null, state :: OptimizationState) = 0.0

"""
    Momentum.Fixed

Fixed momentum scheduler always returns the same value.
"""
type Fixed <: AbstractMomentumScheduler
  momentum :: Float64
end
get_momentum(self :: Fixed, state :: OptimizationState) = self.momentum

"""
    Momentum.NadamScheduler

Nesterov-accelerated adaptive momentum scheduler.

Description in "Incorporating Nesterov Momentum into Adam."
[http://cs229.stanford.edu/proj2015/054_report.pdf]
(http://cs229.stanford.edu/proj2015/054_report.pdf)

``\mu_t = \mu_0 * (1 - \gamma * \alpha^{t * \delta})``.
Here
* ``t`` is the iteration count
* ``\delta``: default `0.004` is scheduler decay,
* ``\gamma``: default `0.5`
* ``\alpha``: default `0.96`
* ``\mu_0``: default `0.99`
"""
type NadamScheduler <: AbstractMomentumScheduler
  mu0 :: Float64
  delta :: Float64
  gamma :: Float64
  alpha :: Float64
end
function NadamScheduler(;mu0::Real=0.99, delta::Real=0.004,
                gamma::Real=0.5, alpha::Real=0.96)
  @assert(0.0 <= delta)
  @assert(0.0 <= alpha <= 1.0)
  @assert(0.0 <= mu0 <= 1.0)
  @assert(0.0 <= gamma <= 1.0)
  NadamScheduler(Float64(mu0), Float64(delta), Float64(gamma), Float64(alpha))
end
get_momentum(self :: NadamScheduler, state :: OptimizationState) =
  self.mu0 * (1.0 - self.gamma*self.alpha^(state.curr_iter * self.delta)),
  self.mu0 * (1.0 - self.gamma*self.alpha^((state.curr_iter + 1) * self.delta))

end # module Momentum
################################################################################
function get_momentum_scheduler(scheduler :: Any, momentum :: Real)
  if isa(scheduler, AbstractMomentumScheduler)
    return scheduler
  elseif momentum == 0
    return Momentum.Null()
  else
    return Momentum.Fixed(momentum)
  end
end

function get_momentum_scheduler(scheduler :: Any,
  another_scheduler :: AbstractMomentumScheduler)

  if isa(scheduler, AbstractMomentumScheduler)
    return scheduler
  else
    return another_scheduler
  end
end

"""
    get_updater(optimizer)

A utility function to create an updater function, that uses its closure to
store all the states needed for each weights.

* `optimizer::AbstractOptimizer`: the underlying optimizer.
"""
function get_updater(optimizer :: AbstractOptimizer)
  states = Dict{Int,Any}()
  function updater(index :: Int, grad :: NDArray, weight :: NDArray)
    if !haskey(states, index)
      states[index] = create_state(optimizer, index, weight)
    end
    update(optimizer, index, weight, grad, states[index])
  end
  return updater
end

"""
    AbstractOptimizerOptions

Base class for all optimizer options.
"""
abstract AbstractOptimizerOptions

"""
    normalized_gradient(opts, state, weight, grad)

* `opts::AbstractOptimizerOptions`: options for the optimizer, should contain the field
`grad_clip` and `weight_decay`.
* `state::OptimizationState`: the current optimization state.
* `weight::NDArray`: the trainable weights.
* `grad::NDArray`: the original gradient of the weights.

   Get the properly normalized gradient (re-scaled and clipped if necessary).
"""
function normalized_gradient(opts::AbstractOptimizerOptions, state::OptimizationState,
                             weight::NDArray, grad::NDArray)
  grad_scale = 1.0 / state.batch_size

  grad = grad_scale * grad
  if opts.grad_clip > 0
    grad = clip(grad, -opts.grad_clip, opts.grad_clip)
  end
  if opts.weight_decay > 0
    @inplace grad += opts.weight_decay * weight
  end

  return grad
end

include("optimizers/sgd.jl")
include("optimizers/adam.jl")
include("optimizers/adagrad.jl")
include("optimizers/adadelta.jl")
include("optimizers/adamax.jl")
include("optimizers/rmsprop.jl")
include("optimizers/nadam.jl")
