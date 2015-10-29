
Models
======

The model API provides convenient high-level interface to do training and predicting on
a network described using the symbolic API.




.. class:: AbstractModel

   The abstract super type of all models in MXNet.jl.




.. class:: FeedForward

   The feedforward model provides convenient interface to train and predict on
   feedforward architectures like multi-layer MLP, ConvNets, etc. There is no
   explicitly handling of *time index*, but it is relatively easy to implement
   unrolled RNN / LSTM under this framework (**TODO**: add example). For models
   that handles sequential data explicitly, please use **TODO**...




.. function:: FeedForward(arch :: Symbol, ctx)

   :param arch: the architecture of the network constructed using the symbolic API.
   :param ctx: the devices on which this model should do computation. It could be a single :class:`Context`
               or a list of :class:`Context` objects. In the latter case, data parallelization will be used
               for training. If no context is provided, the default context ``cpu()`` will be used.




.. function:: init_model(self, initializer; overwrite=false, input_shapes...)

   Initialize the weights in the model.

   This method will be called automatically when training a model. So there is usually no
   need to call this method unless one needs to inspect a model with only randomly initialized
   weights.

   :param FeedForward self: the model to be initialized.
   :param AbstractInitializer initializer: an initializer describing how the weights should be initialized.
   :param Bool overwrite: keyword argument, force initialization even when weights already exists.
   :param input_shapes: the shape of all data and label inputs to this model, given as keyword arguments.
                        For example, ``data=(28,28,1,100), label=(100,)``.




.. function::
   predict(self, data; overwrite=false, callback=nothing)

   Predict using an existing model. The model should be already initialized, or trained or loaded from
   a checkpoint. There is an overloaded function that allows to pass the callback as the first argument,
   so it is possible to do

   .. code-block:: julia

      predict(model, data) do batch_output
        # consume or write batch_output to file
      end

   :param FeedForward self: the model.
   :param AbstractDataProvider data: the data to perform prediction on.
   :param Bool overwrite: an :class:`Executor` is initialized the first time predict is called. The memory
                          allocation of the :class:`Executor` depends on the mini-batch size of the test
                          data provider. If you call predict twice with data provider of the same batch-size,
                          then the executor can be re-used. Otherwise, if ``overwrite`` is false (default),
                          an error will be raised; if ``overwrite`` is set to true, a new :class:`Executor`
                          will be created to replace the old one.

   .. note::

      Prediction is computationally much less costly than training, so the bottleneck sometimes becomes the IO
      for copying mini-batches of data. Since there is no concern about convergence in prediction, it is better
      to set the mini-batch size as large as possible (limited by your device memory) if prediction speed is a
      concern.

      For the same reason, currently prediction will only use the first device even if multiple devices are
      provided to construct the model.

   :seealso: :func:`train`, :func:`fit`, :func:`init_model`, :func:`load_checkpoint`




.. function:: train(model :: FeedForward, ...)

   Alias to :func:`fit`.




.. function:: fit(model :: FeedForward, optimizer, data; kwargs...)

   Train the ``model`` on ``data`` with the ``optimizer``.

   :param FeedForward model: the model to be trained.
   :param AbstractOptimizer optimizer: the optimization algorithm to use.
   :param AbstractDataProvider data: the training data provider.
   :param Int n_epoch: default 10, the number of full data-passes to run.
   :param AbstractDataProvider eval_data: keyword argument, default ``nothing``. The data provider for
          the validation set.
   :param AbstractEvalMetric eval_metric: keyword argument, default ``Accuracy()``. The metric used
          to evaluate the training performance. If ``eval_data`` is provided, the same metric is also
          calculated on the validation set.
   :param kvstore: keyword argument, default ``:local``. The key-value store used to synchronize gradients
          and parameters when multiple devices are used for training.
   :type kvstore: :class:`KVStore` or ``Base.Symbol``
   :param AbstractInitializer initializer: keyword argument, default ``UniformInitializer(0.01)``.
   :param Bool force_init: keyword argument, default false. By default, the random initialization using the
          provided ``initializer`` will be skipped if the model weights already exists, maybe from a previous
          call to :func:`train` or an explicit call to :func:`init_model` or :func:`load_checkpoint`. When
          this option is set, it will always do random initialization at the begining of training.
   :param callbacks: keyword argument, default ``[]``. Callbacks to be invoked at each epoch or mini-batch,
          see :class:`AbstractCallback`.
   :type callbacks: ``Vector{AbstractCallback}``



