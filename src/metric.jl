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

function _update_single_output(metric :: Accuracy, label :: NDArray, pred :: NDArray)
  @nd_as_jl ro=(label,pred) begin
    if ndims(pred) > 2 # Multidimensional case
      # Construct cartesian index
      initial = tuple(fill(1, ndims(pred)-2)...)
      dims = size(pred, (1:ndims(pred)-2)...)
      crange = CartesianRange(CartesianIndex(initial), CartesianIndex(dims))

      for sample in 1:size(label, ndims(label))
        for i in crange
          l_i = sub2ind(dims, i.I...)
          ps = sub(pred, i.I..., :, sample)
          klass = indmax(ps)
          metric.acc_sum += (klass-1) == label[l_i, sample]
          metric.n_sample += 1
        end
      end
    else # 1-dimensional case
      for sample in 1:size(label, 1)
        ps = sub(pred, :, sample)
        klass = indmax(ps)
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


