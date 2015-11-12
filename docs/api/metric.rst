
Evaluation Metrics
==================

Evaluation metrics provide a way to evaluate the performance of a learned model.
This is typically used during training to monitor performance on the validation
set.




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




.. class:: Accuracy

   Multiclass classification accuracy.

   Calculates the mean accuracy per sample for softmax in one dimension.
   For a multi-dimensional softmax the mean accuracy over all dimensions is calculated.




.. class:: MSE

   Mean Squared Error. Todo: add support for multi-dimensional outputs.

   Calculates the mean squared error regression loss in one dimension.



