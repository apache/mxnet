"""
    AbstractEvalMetric

The base class for all evaluation metrics. The sub-types should implement the following
interfaces:

* [`update!`](@ref)
* [`reset!`](@ref)
* [`get`](@ref)
"""
abstract AbstractEvalMetric

"""
    update!(metric, labels, preds)

Update and accumulate metrics.

# Arguments:
* `metric::AbstractEvalMetric`: the metric object.
* `labels::Vector{NDArray}`: the labels from the data provider.
* `preds::Vector{NDArray}`: the outputs (predictions) of the network.
"""
function update!{T <: AbstractEvalMetric}(metric :: T, labels :: Vector{NDArray}, preds :: Vector{NDArray})
  if length(labels) != length(preds)
    Base.warn_once(
      "The number of labels ($(length(labels))) does not correspond to the\
      number of outputs ($(length(preds))). The calculated metric might not be accuracte.")
  end
  for (label, pred) in zip(labels, preds)
    _update_single_output(metric, label, pred)
  end
end

"""
    reset!(metric)

Reset the accumulation counter.
"""
function reset!(metric :: AbstractEvalMetric)
  throw(MethodError(reset!, (typeof(metric),)))
end


import Base: get
"""
    get(metric)

Get the accumulated metrics.

Returns `Vector{Tuple{Base.Symbol, Real}}`, a list of name-value pairs.
For example, `[(:accuracy, 0.9)]`.
"""
function get(metric :: AbstractEvalMetric)
  throw(MethodError(get, (typeof(metric),)))
end

"""
    MultiMetric(metrics::Vector{AbstractEvalMetric})

Combine multiple metrics in one and get a result for all of them.

# Usage
To calculate both mean-squared error [`Accuracy`](@ref) and log-loss [`ACE`](@ref):
```julia
  mx.fit(..., eval_metric = mx.MultiMetric([mx.Accuracy(), mx.ACE()]))
```
"""
type MultiMetric <: mx.AbstractEvalMetric
    metrics :: Vector{mx.AbstractEvalMetric}
end

function update!(metric :: MultiMetric, labels :: Vector{NDArray}, preds :: Vector{NDArray})
    for m in metric.metrics
        update!(m, labels, preds)
    end
    return nothing
end

function reset!(metric :: MultiMetric)
    map(reset!, metric.metrics)
    return nothing
end

function get(metric :: MultiMetric)
    mapreduce(get, append!, metric.metrics)
end

"""
    Accuracy

Multiclass classification accuracy.

Calculates the mean accuracy per sample for softmax in one dimension.
For a multi-dimensional softmax the mean accuracy over all dimensions is calculated.
"""
type Accuracy <: AbstractEvalMetric
  acc_sum  :: Float64
  n_sample :: Int

  Accuracy() = new(0.0, 0)
end

function _update_single_output(metric :: Accuracy, label :: NDArray, pred :: NDArray)
  @nd_as_jl ro=(label,pred) begin
    # Samples are stored in the last dimension
    @assert size(label, ndims(label)) == size(pred, ndims(pred))

    if ndims(pred) == 4 # Multidimensional case
      # Reshape label to be of the same shape as pred.
      # Except for the third dimension where the predictions are stored.
      labels = reshape(label, size(pred, 1, 2)..., 1, size(pred, 4))

      for sample in 1:size(labels, 4)
        for j in 1:size(labels, 2)
          for i in 1:size(labels, 1)
            label = labels[i, j, 1, sample]
            klasses = view(pred, i, j, :, sample)
            klass = indmax(klasses) - 1 # Classes start at 0...k-1

            metric.acc_sum += klass == label
            metric.n_sample += 1
          end
        end
      end
    elseif ndims(pred) == 2 # 1-dimensional case
      for sample in 1:size(label, 1)
        klass = indmax(view(pred, :, sample)) - 1
        metric.acc_sum += klass == label[sample]
        metric.n_sample += 1
      end
    else
      error("Can't handle prediction with dimensions $(ndims(pred)).")
    end
  end
end

function get(metric :: Accuracy)
  return [(:accuracy, metric.acc_sum / metric.n_sample)]
end

function reset!(metric :: Accuracy)
  metric.acc_sum  = 0.0
  metric.n_sample = 0
end

"""
    MSE

Mean Squared Error. TODO: add support for multi-dimensional outputs.

Calculates the mean squared error regression loss in one dimension.
"""

type MSE <: AbstractEvalMetric
  mse_sum  :: Float64
  n_sample :: Int

  MSE() = new(0.0, 0)
end

function _update_single_output(metric :: MSE, label :: NDArray, pred :: NDArray)
  label = copy(label)
  pred  = copy(pred)

  n_sample = size(pred)[end]
  metric.n_sample += n_sample

  for i = 1:n_sample
    metric.mse_sum += (label[i] - pred[i])^2
  end
end

function get(metric :: MSE)
  return [(:MSE, metric.mse_sum / metric.n_sample)]
end

function reset!(metric :: MSE)
  metric.mse_sum  = 0.0
  metric.n_sample = 0
end

"""
    ACE

Calculates the averaged cross-entropy (logloss) for classification.
"""
type ACE <: AbstractEvalMetric
  ace_sum  :: Float64
  n_sample :: Int

  ACE() = new(0.0, 0)
end

function get(metric :: ACE)
  return [(:ACE, - metric.ace_sum / metric.n_sample)]
end

function reset!(metric :: ACE)
  metric.ace_sum = 0.0
  metric.n_sample = 0
end

function _update_single_output(metric :: ACE, label :: NDArray, pred :: NDArray)
  @nd_as_jl ro=(label,pred) begin
    # Samples are stored in the last dimension
    @assert size(label, ndims(label)) == size(pred, ndims(pred))
    if ndims(pred) == 4
      labels = reshape(label, size(pred, 1, 2)..., 1, size(pred, 4))
      for sample in 1:size(labels, 4)
        for j in 1:size(labels, 2)
          for i in 1:size(labels, 1)
            # Cross-entropy reduces to -(ln(p_1)*0 + ln(p_2)*1) for classification
            # Since we can only target labels right now this is the only thing we can do.
            target = Int(labels[i, j, 1, sample]) + 1 # klasses are 0...k-1 => julia indexing
            p_k = pred[i, j, target, sample]
            metric.ace_sum += log(p_k)
            metric.n_sample += 1
          end
        end
      end
    elseif ndims(pred) == 2 # 1-dimensional case
      for sample in 1:size(label, 1)
        target = Int(label[sample]) + 1    # 0-based indexing => 1-based indexing
        p_k = pred[target, sample]
        metric.ace_sum += log(p_k)
        metric.n_sample += 1
      end
    else
      error("Can't handle prediction with dimensions $(ndims(pred)).")
    end
  end
end

"""
    MultiACE

Calculates the averaged cross-entropy per class and overall (see [`ACE`](@ref)).
This can be used to quantify the influence of different classes on the overall loss.
"""
type MultiACE <: AbstractEvalMetric
  aces  :: Vector{Float64}
  counts :: Vector{Int}

  MultiACE(nclasses) = new(Base.zeros(nclasses), Base.zeros(Int, nclasses))
end

function get(metric :: MultiACE)
  aces = [(Symbol("ACE_$(i-0)"), - metric.aces[i] / metric.counts[i]) for i in 1:length(metric.aces)]
  push!(aces, (:ACE, - Base.sum(metric.aces) / Base.sum(metric.counts)))
  return aces
end

function reset!(metric :: MultiACE)
  metric.aces = Base.zero(metric.aces)
  metric.counts = Base.zero(metric.counts)
end

function _update_single_output(metric :: MultiACE, label :: NDArray, pred :: NDArray)
  @nd_as_jl ro=(label,pred) begin
    # Samples are stored in the last dimension
    @assert size(label, ndims(label)) == size(pred, ndims(pred))

    if ndims(pred) == 4
      labels = reshape(label, size(pred, 1, 2)..., 1, size(pred, 4))
      for sample in 1:size(labels, 4)
        for j in 1:size(labels, 2)
          for i in 1:size(labels, 1)
            # Cross-entropy reduces to -(ln(p_1)*0 + ln(p_2)*1) for classification
            # Since we can only target labels right now this is the only thing we can do.
            target = Int(labels[i, j, 1, sample]) + 1 # klasses are 0...k-1 => julia indexing
            p_k = pred[i, j, target, sample]

            metric.aces[target] += log(p_k)
            metric.counts[target] += 1
          end
        end
      end
    elseif ndims(pred) == 2
      for sample in 1:size(label, 1)
        target = Int(label[sample]) + 1
        p_k = pred[target, sample]
        metric.aces[target] += log(p_k)
        metric.counts[target] += 1
      end
    else
      error("Can't handle prediction with dimensions $(ndims(pred)).")
    end
  end
end

