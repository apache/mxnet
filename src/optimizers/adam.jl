
@defstruct ADAMOptions Any (
  (lr           :: Real = 0.001, lr > 0),
  (lr_decay     :: Real = 1.0, lr_decay > 0),
  (beta1        :: Real = 0.9,  beta1 > 0),
  (beta2        :: Real = 0.999,  beta2 > 0),
  (epsilon      :: Real = 1e-8, epsilon > 0),
  (grad_scale   :: Real = 1.0, grad_scale >= 0),
  (grad_clip    :: Real = 0, grad_clip >= 0)
)


type ADAM <: AbstractOptimizer
  iter       :: Int
  batch_size :: Int
  opts       :: ADAMOptions

  function ADAM(; kwargs...)
    opts = ADAMOptions(;kwargs...)
    
    new(0, 0, opts)
  end
end

type ADAMState
  current_lr :: Float64  # current learning rate
  mt :: NDArray
  vt :: NDArray
  beta1Power :: Float64
  beta2Power :: Float64
end

function create_state(self :: ADAM, index :: Int, weight :: NDArray)
  return ADAMState( self.opts.lr, 
                    zeros(size(weight), context(weight)), 
                    zeros(size(weight), context(weight)),
                    self.opts.beta1,
                    self.opts.beta2 )
end

function update(self :: ADAM, index :: Int, weight :: NDArray, grad :: NDArray, state :: ADAMState)
  lr = state.current_lr
  grad_scale = self.opts.grad_scale / self.batch_size

  grad = grad_scale * grad
  if self.opts.grad_clip > 0
    grad = clip(grad, -self.opts.grad_clip, self.opts.grad_clip)
  end

  state.mt = self.opts.beta1 * state.mt + (1 - self.opts.beta1) * grad
  state.vt = self.opts.beta2 * state.vt + (1 - self.opts.beta2) * (grad .* grad)

  mt = state.mt / (1 - state.beta1Power)
  vt = state.vt / (1 - state.beta2Power)

  #@show state.beta1Power,state.beta2Power

  state.beta1Power *= self.opts.beta1
  state.beta2Power *= self.opts.beta2

  @inplace weight .+= -lr * mt ./ (sqrt(vt) + self.opts.epsilon)

  state.current_lr *= self.opts.lr_decay
end
