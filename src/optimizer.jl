abstract AbstractOptimizer

abstract AbstractLearningRateScheduler
abstract AbstractMomentumScheduler

type FixedLearningRateScheduler <: AbstractLearningRateScheduler
  learning_rate :: Float64
end
get_learning_rate(self :: FixedLearningRateScheduler, iter :: Int) = self.learning_rate

type NullMomentumScheduler <: AbstractMomentumScheduler
end
get_momentum(self :: NullMomentumScheduler, iter :: Int) = 0.0

type FixedMomentumScheduler <: AbstractMomentumScheduler
  momentum :: Float64
end
get_momentum(self :: FixedMomentumScheduler, iter :: Int) = self.momentum

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
