# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
    AbstractEvalMetric

The base class for all evaluation metrics. The sub-types should implement the following
interfaces:

* [`update!`](@ref)
* [`reset!`](@ref)
* [`get`](@ref)
"""
abstract type AbstractEvalMetric end

"""
    hasNDArraySupport(metric) -> Val{true/false}

Trait for `_update_single_output` should return `Val{true}() if metric can handle `NDArray`
directly and `Val{false}()` if requires `Array`. Metric that work with NDArrays can be
async, while native Julia arrays require that we copy the output of the network, which is
a blocking operation.
"""
hasNDArraySupport(::AbstractEvalMetric) = Val{true}()

"""
    update!(metric, labels, preds)

Update and accumulate metrics.

# Arguments:
* `metric::AbstractEvalMetric`: the metric object.
* `labels::Vector{NDArray}`: the labels from the data provider.
* `preds::Vector{NDArray}`: the outputs (predictions) of the network.
"""
update!(metric::T, labels::VecOfNDArray, preds::VecOfNDArray) where T<:AbstractEvalMetric =
  _update!(metric, labels, preds, hasNDArraySupport(metric))

function _update!(metric::T, labels::VecOfNDArray, preds::VecOfNDArray,
                  ::Val{true}) where T<:AbstractEvalMetric
  if length(labels) != length(preds)
    @warn(
      "The number of labels ($(length(labels))) does not correspond to the " *
      "number of outputs ($(length(preds))). The calculated metric might not be accuracte.",
      maxlog = 1)
  end
  for (label, pred) in zip(labels, preds)
    _update_single_output(metric, label, pred)
  end
end

function _update!(metric::T, labels::VecOfNDArray, preds::VecOfNDArray,
                  ::Val{false}) where {T<:AbstractEvalMetric}
  if length(labels) != length(preds)
    @warn(
      "The number of labels ($(length(labels))) does not correspond to the " *
      "number of outputs ($(length(preds))). The calculated metric might not be accuracte.",
      maxlog = 1)
  end
  for (label, pred) in zip(labels, preds)
     @nd_as_jl ro=(label, pred) begin
       # This is a dynamic dispatch since the conversion from NDArray to
       # Array is not type-stable.
      _update_single_output(metric, label, pred)
    end
  end
end

"""
    reset!(metric)

Reset the accumulation counter.
"""
reset!(metric::AbstractEvalMetric) = throw(MethodError(reset!, (typeof(metric),)))


import Base: get
"""
    get(metric)

Get the accumulated metrics.

Returns `Vector{Tuple{Base.Symbol, Real}}`, a list of name-value pairs.
For example, `[(:accuracy, 0.9)]`.
"""
get(metric::AbstractEvalMetric) = throw(MethodError(get, (typeof(metric),)))

"""
    NullMetric()

A metric that calculates nothing. Can be used to ignore an output during training.
"""
mutable struct NullMetric <: mx.AbstractEvalMetric
end

update!(metric::NullMetric, labels::VecOfNDArray, preds::VecOfNDArray) = nothing

reset!(metric::NullMetric) = nothing

get(metric::NullMetric) = Tuple{Symbol, Float64}[]

"""
    MultiMetric(metrics::Vector{AbstractEvalMetric})

Combine multiple metrics in one and get a result for all of them.

# Usage
To calculate both mean-squared error [`Accuracy`](@ref) and log-loss [`ACE`](@ref):
```julia
  mx.fit(..., eval_metric = mx.MultiMetric([mx.Accuracy(), mx.ACE()]))
```
"""
mutable struct MultiMetric <: AbstractEvalMetric
  metrics :: Vector{mx.AbstractEvalMetric}
end

function update!(metric :: MultiMetric, labels :: Vector{<:NDArray}, preds :: Vector{<:NDArray})
  for m in metric.metrics
    update!(m, labels, preds)
  end
  nothing
end

function reset!(metric :: MultiMetric)
  map(reset!, metric.metrics)
  nothing
end

get(metric::MultiMetric) = mapreduce(get, append!, metric.metrics)

"""
    SeqMetric(metrics::Vector{AbstractEvalMetric})

Apply a different metric to each output. This is especially useful for `mx.Group`.

# Usage
Calculate accuracy [`Accuracy`](@ref) for the first output
and log-loss [`ACE`](@ref) for the second output:
```julia
  mx.fit(..., eval_metric = mx.SeqMetric([mx.Accuracy(), mx.ACE()]))
```
"""
mutable struct SeqMetric <: AbstractEvalMetric
  metrics :: Vector{AbstractEvalMetric}
end

function update!(metric::SeqMetric, labels::VecOfNDArray, preds::VecOfNDArray)
  @assert length(metric.metrics) == length(labels)
  @assert length(metric.metrics) == length(preds)
  for (m, l, p) in zip(metric.metrics, labels, preds)
    update!(m, [l], [p])
  end
  nothing
end

function reset!(metric::SeqMetric)
  map(reset!, metric.metrics)
  nothing
end

get(metric::SeqMetric) = mapreduce(get, append!, metric.metrics)

"""
    Accuracy

Multiclass classification accuracy.

Calculates the mean accuracy per sample for softmax in one dimension.
For a multi-dimensional softmax the mean accuracy over all dimensions is calculated.
"""
mutable struct Accuracy <: AbstractEvalMetric
  acc_sum  :: Float64
  n_sample :: Int

  Accuracy() = new(0.0, 0)
end

hasNDArraySupport(::Accuracy) = Val{false}()

function _update_single_output(metric::Accuracy, label::Array, pred::Array)
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
          klass = argmax(klasses) - 1 # Classes start at 0...k-1

          metric.acc_sum += klass == label
          metric.n_sample += 1
        end
      end
    end
  elseif ndims(pred) == 2 # 1-dimensional case
    for sample in 1:size(label, 1)
      klass = argmax(view(pred, :, sample)) - 1
      metric.acc_sum += klass == label[sample]
      metric.n_sample += 1
    end
  else
    error("Can't handle prediction with dimensions $(ndims(pred)).")
  end
end

get(metric::Accuracy) = [(:accuracy, metric.acc_sum / metric.n_sample)]

function reset!(metric :: Accuracy)
  metric.acc_sum  = 0.0
  metric.n_sample = 0
end

"""
    MSE

Mean Squared Error.

Calculates the mean squared error regression loss.
Requires that label and prediction have the same shape.
"""
mutable struct MSE{N} <: AbstractEvalMetric
  mse_sum  :: Vector{NDArray{MX_float,N}}
  n_sample :: Int

  MSE{N}() where {N} = new(Vector{NDArray{MX_float,N}}(), 0)
end

MSE() = MSE{1}()  # backward compat?

hasNDArraySupport(::MSE) = Val{true}()

function _update_single_output(metric::MSE, label::NDArray{T,N},
                               pred::NDArray{T,N}) where {T,N}
  @assert size(label) == size(pred)
  metric.n_sample += length(label)
  mse_sum = mx.sum((label .- pred).^2)
  push!(metric.mse_sum, mse_sum)
  nothing
end

function get(metric::MSE)
  # Delay copy until last possible moment
  mse_sum = mapreduce(nda->copy(nda)[1], +, 0.0, metric.mse_sum)
  [(:MSE, mse_sum / metric.n_sample)]
end

function reset!(metric::MSE{N}) where N
  metric.mse_sum = Vector{NDArray{Float32,N}}()
  metric.n_sample = 0
end

@doc doc"""
    NMSE

Normalized Mean Squared Error

```math
\sum_i (\frac{label_i - pred_i}{label_i})^2
```

Note that there are various ways to do the *normalization*.
It depends on your own context. Please judge the problem setting you have
first. If the current implementation do not suitable for you,
feel free to file it on GitHub.

Let me show you a use case of this kind of normalization:

Bob is training a network for option pricing. The option pricing problem is
a regression problem (pirce predicting). There are lots of option contracts
on same target stock but different strike price.
For example, there is a stock `S`; it's market price is 1000.
And, there are two call option contracts with different strike price.
Assume Bob obtains the outcome as following table:

```
+--------+----------------+----------------+--------------+
|        | Strike Price   | Market Price   | Pred Price   |
+--------+----------------+----------------+--------------+
| Op 1   | 1500           |  100           | 80           |
+--------+----------------+----------------+--------------+
| Op 2   | 500            |  10            | 8            |
+--------+----------------+----------------+--------------+
```

Now, obviously, Bob will calculate the normalized MSE as:

```math
    (\frac{100 - 80}{100})^2
    \text{ vs }
    (\frac{10 - 8}{10}) ^2
```

Both of the pred prices got the same degree of error.

For more discussion about normalized MSE, please see
[#211](https://github.com/dmlc/MXNet.jl/pull/211) also.

"""
mutable struct NMSE <: AbstractEvalMetric
  nmse_sum  :: Float64
  n_sample :: Int

  NMSE() = new(0.0, 0)
end

hasNDArraySupport(::NMSE) = Val{false}()

function _update_single_output(metric::NMSE, label::Array, pred::Array)
  n_sample = size(pred)[end]
  metric.n_sample += n_sample

  for i = 1:n_sample
    if label[i] == 0.0f0  # in case of batch padding
        continue
    end

    metric.nmse_sum += ((label[i] - pred[i]) / label[i])^2
  end
end

get(metric::NMSE) = [(:NMSE, metric.nmse_sum / metric.n_sample)]

function reset!(metric::NMSE)
  metric.nmse_sum = 0.0
  metric.n_sample = 0
end

"""
    ACE

Calculates the averaged cross-entropy (logloss) for classification.

# Arguments:
* `eps::Float64`: Prevents returning `Inf` if `p = 0`.
"""
mutable struct ACE <: AbstractEvalMetric
  ace_sum  :: Float64
  n_sample :: Int
  eps :: Float64

  ACE(eps=1.0e-8) = new(0.0, 0, eps)
end

get(metric::ACE) = [(:ACE, - metric.ace_sum / metric.n_sample)]

function reset!(metric::ACE)
  metric.ace_sum = 0.0
  metric.n_sample = 0
end

hasNDArraySupport(::ACE) = Val{false}()

function _update_single_output(metric :: ACE, label :: Array{T}, pred :: Array{T}) where T
  eps = convert(T, metric.eps)
  # Samples are stored in the last dimension
  @assert size(label, ndims(label)) == size(pred, ndims(pred))
  if size(label) == size(pred) # simply calculate the cross entropy of the probabilities
    for (q, p) in zip(pred, label)
      # p == true probability
      # q == "unnatural" probability
      metric.ace_sum += p * log(q + eps)
      metric.n_sample += 1
    end
  elseif ndims(pred) == 4
    labels = reshape(label, size(pred, 1, 2)..., 1, size(pred, 4))
    for sample in 1:size(labels, 4)
      for j in 1:size(labels, 2)
        for i in 1:size(labels, 1)
          # Cross-entropy reduces to -(ln(p_1)*0 + ln(p_2)*1) for classification
          # Since we can only target labels right now this is the only thing we can do.
          target = Int(labels[i, j, 1, sample]) + 1 # klasses are 0...k-1 => julia indexing
          p_k = pred[i, j, target, sample]
          metric.ace_sum += log(p_k + eps)
          metric.n_sample += 1
        end
      end
    end
  elseif ndims(pred) == 2 # 1-dimensional case
    for sample in 1:size(label, 1)
      target = Int(label[sample]) + 1    # 0-based indexing => 1-based indexing
      p_k = pred[target, sample]
      metric.ace_sum += log(p_k +eps)
      metric.n_sample += 1
    end
  else
    error("Can't handle prediction with dimensions $(ndims(pred)).")
  end
end

"""
    MultiACE

Calculates the averaged cross-entropy per class and overall (see [`ACE`](@ref)).
This can be used to quantify the influence of different classes on the overall loss.
"""
mutable struct MultiACE <: AbstractEvalMetric
  aces  :: Vector{Float64}
  counts :: Vector{Int}
  eps :: Float64

  MultiACE(nclasses, eps=1.0e-8) = new(Base.zeros(nclasses), Base.zeros(Int, nclasses), eps)
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

hasNDArraySupport(::MultiACE) = Val{false}()

function _update_single_output(metric :: MultiACE, label :: Array{T}, pred :: Array{T}) where T
  eps = convert(T, metric.eps)
  # Samples are stored in the last dimension
  @assert size(label, ndims(label)) == size(pred, ndims(pred))
  @assert size(metric.aces) == size(metric.counts)
  if size(label) == size(pred) # simply calculate the cross entropy of the probabilities
    for k in 1:length(metric.aces)
      kpred  = view(pred,  ntuple(d->:, ndims(pred)  - 2)..., k, :)
      klabel = view(label, ntuple(d->:, ndims(label) - 2)..., k, :)
      for (q, p) in zip(kpred, klabel)
        # p == true probability
        # q == "unnatural" probability
        metric.aces[k] += p * log(q + eps)
        metric.counts[k] += 1
      end
    end
  elseif ndims(pred) == 4
    labels = reshape(label, size(pred, 1, 2)..., 1, size(pred, 4))
    for sample in 1:size(labels, 4)
      for j in 1:size(labels, 2)
        for i in 1:size(labels, 1)
          # Cross-entropy reduces to -(ln(p_1)*0 + ln(p_2)*1) for classification
          # Since we can only target labels right now this is the only thing we can do.
          target = Int(labels[i, j, 1, sample]) + 1 # klasses are 0...k-1 => julia indexing
          p_k = pred[i, j, target, sample]

          metric.aces[target] += log(p_k + eps)
          metric.counts[target] += 1
        end
      end
    end
  elseif ndims(pred) == 2
    for sample in 1:size(label, 1)
      target = Int(label[sample]) + 1
      p_k = pred[target, sample]
      metric.aces[target] += log(p_k + eps)
      metric.counts[target] += 1
    end
  else
    error("Can't handle prediction with dimensions $(ndims(pred)).")
  end
end
