
Neural Networks Factory
=======================

Neural network factory provide convenient helper functions to define
common neural networks.




.. function:: MLP(input, spec)

   Construct a multi-layer perceptron.

   :param SymbolicNode input: the input to the mlp.
   :param spec: the mlp specification, a list of hidden dimensions. For example,
          ``[128, (512, :sigmoid), 10]``. The number in the list indicate the
          number of hidden units in each layer. A tuple could be used to specify
          the activation of each layer. Otherwise, the default activation will
          be used (except for the last layer).
   :param Base.Symbol hidden_activation: keyword argument, default ``:relu``, indicating
          the default activation for hidden layers. The specification here could be overwritten
          by layer-wise specification in the ``spec`` argument. Also activation is not
          applied to the last, i.e. the prediction layer.
   :param prefix: keyword argument, default ``gensym()``, used as the prefix to
          name the constructed layers.



