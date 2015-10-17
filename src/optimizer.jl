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

type SGD <: AbstractOptimizer
  iter          :: Int

  lr_scheduler  :: AbstractLearningRateScheduler
  mom_scheduler :: AbstractMomentumScheduler
  weight_decay  :: Float64
  grad_scale    :: Float64
  grad_clip     :: Float64
  inv_batch_size:: Float64

  function SGD(;lr_scheduler::AbstractLearningRateScheduler=FixedLearningRateScheduler(0.01),
               mom_scheduler::AbstractMomentumScheduler=NullMomentumScheduler(),
               weight_decay::Float64=0.0001,
               grad_scale::Float64=1.0,
               grad_clip::Float64=0.0)
    new(0, lr_scheduler, mom_scheduler, weight_decay, grad_scale, grad_clip, 1.0)
  end
end

function create_state(self :: SGD, index :: Int, weight :: NDArray)
  if isa(self.mom_scheduler, NullMomentumScheduler)
    return nothing
  else
    return zeros(size(weight), context(weight))
  end
end

function update(self :: SGD, index :: Int, weight :: NDArray, grad :: NDArray, state :: Union{Void, NDArray})
  lr = get_learning_rate(self.lr_scheduler, self.iter)
  grad_scale = self.grad_scale * self.inv_batch_size

  if isa(state, Void)
    @inplace weight += -lr * (grad_scale * grad + self.weight_decay * weight)
  else
    mom = state :: NDArray
    coef = get_momentum(self.mom_scheduler, self.iter)
    @inplace mom .*= coef
    if self.clip_gradient > 0
      # TODO:
    else
      @inplace mom += -lr * (grad_scale * grad + self.weight_decay * weight)
    end
    @inplace weight += mom
  end
end


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
