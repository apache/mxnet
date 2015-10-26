"Abstract type of callback functions used in training"
abstract AbstractCallback

"Abstract type of callbacks to be called every mini-batch"
abstract AbstractIterationCallback <: AbstractCallback

"Abstract type of callbacks to be called every epoch"
abstract AbstractEpochCallback <: AbstractCallback

type CallbackParams
  batch_size :: Int
  curr_epoch :: Int
  curr_iter  :: Int
end
CallbackParams(batch_size::Int) = CallbackParams(batch_size, 0, 0)

type IterationCallback <: AbstractIterationCallback
  frequency :: Int
  call_on_0 :: Bool
  callback  :: Function
end

function every_n_iter(callback :: Function, n :: Int; call_on_0 :: Bool = false)
  IterationCallback(n, call_on_0, callback)
end
function Base.call(cb :: IterationCallback, param :: CallbackParams)
  if param.curr_iter == 0
    if cb.call_on_0
      cb.callback(param)
    end
  elseif param.curr_iter % cb.frequency == 0
    cb.callback(param)
  end
end

function speedometer(;frequency::Int=50)
  cl_tic = 0
  every_n_iter(frequency, call_on_0=true) do param :: CallbackParams
    if param.curr_iter == 0
      # reset counter
      cl_tic = time()
    else
      speed = frequency * param.batch_size / (time() - cl_tic)
      info(format("Speed: {1:>6.2f} samples/sec", speed))
      cl_tic = time()
    end
  end
end


type EpochCallback <: AbstractEpochCallback
  frequency :: Int
  call_on_0 :: Bool
  callback  :: Function
end

function every_n_epoch(callback :: Function, n :: Int; call_on_0 :: Bool = false)
  EpochCallback(n, call_on_0, callback)
end
function Base.call(cb :: EpochCallback, model :: Any, param :: CallbackParams)
  if param.curr_epoch == 0
    if cb.call_on_0
      cb.callback(model, param)
    end
  elseif param.curr_epoch % cb.frequency == 0
    cb.callback(model, param)
  end
end

function do_checkpoint(prefix::AbstractString; frequency::Int=1, save_epoch_0=false)
  mkpath(dirname(prefix))
  every_n_epoch(frequency, call_on_0=save_epoch_0) do model, param
    save_checkpoint(model, prefix, param)
  end
end
