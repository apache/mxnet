#=doc
Optimizers
==========

Common interfaces
-----------------
=#


#=doc
.. class:: AbstractOptimizer

   Base type for all optimizers.
=#
abstract AbstractOptimizer

#=doc
.. class:: AbstractLearningRateScheduler

   Base type for all learning rate scheduler.
=#
abstract AbstractLearningRateScheduler

#=doc
.. class:: AbstractMomentumScheduler

   Base type for all momentum scheduler.
=#
abstract AbstractMomentumScheduler



#=doc
.. class:: OptimizationState

   .. attribute:: batch_size

      The size of the mini-batch used in stochastic training.

   .. attribute:: curr_epoch

      The current epoch count. Epoch 0 means no training yet, during the first
      pass through the data, the epoch will be 1; during the second pass, the
      epoch count will be 1, and so on.

   .. attribute:: curr_batch

      The current mini-batch count. The batch count is reset during every epoch.
      The batch count 0 means the beginning of each epoch, with no mini-batch
      seen yet. During the first mini-batch, the mini-batch count will be 1.

   .. attribute:: curr_iter

      The current iteration count. One iteration corresponds to one mini-batch,
      but unlike the mini-batch count, the iteration count does **not** reset
      in each epoch. So it track the *total* number of mini-batches seen so far.
=#
type OptimizationState
  batch_size :: Int
  curr_epoch :: Int
  curr_batch :: Int
  curr_iter  :: Int
end
OptimizationState(batch_size::Int) = OptimizationState(batch_size, 0, 0, 0)


#=doc
.. function:: get_learning_rate(scheduler, state)

   :param AbstractLearningRateScheduler scheduler: a learning rate scheduler.
   :param OptimizationState state: the current state about epoch, mini-batch and iteration count.
   :return: the current learning rate.
=#
function get_learning_rate
end

################################################################################
# The learning rate module
module LearningRate
import ..mx: AbstractLearningRateScheduler, OptimizationState, get_learning_rate

#=doc
.. class:: LearningRate.Fixed

   Fixed learning rate scheduler always return the same learning rate.
=#
type Fixed <: AbstractLearningRateScheduler
  learning_rate :: Float64
end
get_learning_rate(self :: Fixed, state :: OptimizationState) = self.learning_rate

#=doc
.. class:: LearningRate.Exp

   :math:`\eta_t = \eta_0\gamma^t`. Here :math:`t` is the epoch count, or the iteration
   count if ``decay_on_iteration`` is set to true.
=#
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
#=doc
.. class:: LearningRate.Inv

   :math:`\eta_t = \eta_0 * (1 + \gamma * t)^(-power)`.
   Here :math:`t` is the epoch count, or the iteration count if ``decay_on_iteration``
   is set to true.
=#
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


#=doc
.. function:: get_momentum(scheduler, state)

   :param AbstractMomentumScheduler scheduler: the momentum scheduler.
   :param OptimizationState state: the state about current epoch, mini-batch and iteration count.
   :return: the current momentum.
=#
function get_momentum
end


################################################################################
# The Momentum module
module Momentum
import ..mx: AbstractMomentumScheduler, OptimizationState, get_momentum

#=doc
.. class:: Momentum.Null

   The null momentum scheduler always returns 0 for momentum. It is also used to
   explicitly indicate momentum should not be used.
=#
type Null <: AbstractMomentumScheduler
end
get_momentum(self :: Null, state :: OptimizationState) = 0.0

#=doc
.. class:: Momentum.Fixed

  Fixed momentum scheduler always returns the same value.
=#
type Fixed <: AbstractMomentumScheduler
  momentum :: Float64
end
get_momentum(self :: Fixed, state :: OptimizationState) = self.momentum
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


#=doc
.. function:: get_updater(optimizer)

   :param AbstractOptimizer optimizer: the underlying optimizer.

   A utility function to create an updater function, that uses its closure to
   store all the states needed for each weights.
=#
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

################################################################################
#=doc
Built-in optimizers
-------------------
=#

#=doc
.. class:: AbstractOptimizerOptions

   Base class for all optimizer options.
=#
abstract AbstractOptimizerOptions

#=doc
.. function:: normalized_gradient(opts, state, grad)

   :param AbstractOptimizerOptions opts: options for the optimizer, should contain the field
          ``grad_scale``, ``grad_clip`` and ``weight_decay``.
   :param OptimizationState state: the current optimization state.
   :param NDArray weight: the trainable weights.
   :param NDArray grad: the original gradient of the weights.

   Get the properly normalized gradient (re-scaled and clipped if necessary).
=#
function normalized_gradient(opts::AbstractOptimizerOptions, state::OptimizationState,
                             weight::NDArray, grad::NDArray)
  grad_scale = 1.0 / state.batch_size

  grad = grad_scale * grad
  if opts.grad_clip > 0
    grad = clip(grad, -opts.grad_clip, opts.grad_clip)
  end
  @inplace grad += opts.weight_decay * weight

  return grad
end

include("optimizers/sgd.jl")
include("optimizers/adam.jl")
