#=doc
Optimizers
==========
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

end # module LearningRate
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

include("optimizers/sgd.jl")
include("optimizers/adam.jl")
