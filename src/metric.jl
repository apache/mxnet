#=doc
Evaluation Metrics
==================

Evaluation metrics provide a way to evaluate the performance of a learned model.
This is typically used during training to monitor performance on the validation
set.
=#

#=doc
.. class:: AbstractEvalMetric

   The base class for all evaluation metrics. The sub-types should implement the following
   interfaces.

   .. function:: update!(metric, labels, preds)

      Update and accumulate metrics.

      :param AbstractEvalMetric metric: the metric object.
      :param labels: the labels from the data provider.
      :type labels: Vector{NDArray}
      :param preds: the outputs (predictions) of the network.
      :type preds: Vector{NDArray}

   .. function:: reset!(metric)

      Reset the accumulation counter.

   .. function:: get(metric)

      Get the accumulated metrics.

      :return: ``Vector{Tuple{Base.Symbol, Real}}``, a list of name-value pairs. For
               example, ``[(:accuracy, 0.9)]``.
=#
abstract AbstractEvalMetric

#=doc
.. class:: Accuracy

   Multiclass classification accuracy.
=#
type Accuracy <: AbstractEvalMetric
  acc_sum  :: Float64
  n_sample :: Int

  Accuracy() = new(0.0, 0)
end

"""
Implementation taken from findmax in Julia base.
Searches for the maximum value in p_dim of a.
I and n are values for the other dimensions.
"""
function _indmax(a, I, p_dim, n)
  m = a[I..., 1, n]
  mi = 1
  for i in 2:size(a, p_dim)
    ai = a[I..., i, n]
    if ai > m || m!=m
      m = ai
      mi = i
    end
  end
  return mi
end

function _update_single_output(metric :: Accuracy, label :: NDArray, pred :: NDArray)
  @nd_as_jl ro=(label,pred) begin
    if ndims(pred) > 2 # Multidimensional case
      # Construct cartesian index
      p_dim = ndims(pred)-1
      initial = tuple(fill(1,p_dim-1)...)
      dims = size(pred, (1:p_dim-1)...)
      crange = CartesianRange(CartesianIndex(initial), CartesianIndex(dims))

      for sample in 1:size(label, ndims(label))
        for i in crange
          l_i = sub2ind(dims, i.I...)
          klass = _indmax(pred, i.I, p_dim, sample)
          metric.acc_sum += (klass-1) == label[l_i, sample]
          metric.n_sample += 1
        end
      end
    else # 1-dimensional case
      for sample in 1:size(label, 1)
        klass = indmax(pred[:, sample])
        metric.acc_sum += (klass-1) == label[sample]
        metric.n_sample += 1
      end
    end
  end
end

function update!(metric :: Accuracy, labels :: Vector{NDArray}, preds :: Vector{NDArray})
  @assert length(labels) == length(preds)
  for i = 1:length(labels)
    _update_single_output(metric, labels[i], preds[i])
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


