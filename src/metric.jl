abstract AbstractEvalMetric

type Accuracy <: AbstractEvalMetric
  acc_sum  :: Float64
  n_sample :: Int

  Accuracy() = new(0.0, 0)
end

function _update_single_output(metric :: Accuracy, labels :: NDArray, pred :: NDArray)
  label = copy(label)
  pred  = copy(pred)

  n_sample = size(pred)[end]
  metric.n_sample += n_sample
  for i = 1:n_sample
    klass = indmax(sub(pred,:,i))
    metric.acc_sum += (klass-1) == label[i]
  end
end

function update!(metric :: Accuracy, labels :: Vector{NDArray}, preds :: Vector{NDArray})
  @assert length(labels) == length(preds)
  for i = 1:length(labels)
    _update_single_output(labels[i], preds[i])
  end
end

import Base: get
function get(metric :: Accuracy)
  return [(:accuracy, metric.acc_sum / metric.n_sample)]
end

function reset!(metric :: Accuracy)
  metric.acc_sum  = 0.0
  metric.n_sample = 0
end
