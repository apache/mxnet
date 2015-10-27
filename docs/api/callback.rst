
Callbacks in training
=====================




.. class:: AbstractCallback

   Abstract type of callback functions used in training.




.. class:: AbstractIterationCallback

   Abstract type of callbacks to be called every mini-batch.




.. class:: AbstractEpochCallback

   Abstract type of callbacks to be called every epoch.




.. function:: every_n_iter(callback :: Function, n :: Int; call_on_0 = false)

   A convenient function to construct a callback that runs every ``n`` mini-batches.

   :param Int call_on_0: keyword argument, default false. Unless set, the callback
          will **not** be run on iteration 0.

   For example, the :func:`speedometer` callback is defined as

   .. code-block:: julia

      every_n_iter(frequency, call_on_0=true) do param :: CallbackParams
        if param.curr_iter == 0
          # reset timer
        else
          # compute and print speed
        end
      end

   :seealso: :func:`every_n_epoch`, :func:`speedometer`.




.. function:: speedometer(; frequency=50)

   Create an :class:`AbstractIterationCallback` that measure the training speed
   (number of samples processed per second) every k mini-batches.

   :param Int frequency: keyword argument, default 50. The frequency (number of
          min-batches) to measure and report the speed.




.. function:: every_n_epoch(callback :: Function, n :: Int; call_on_0 = false)

   A convenient function to construct a callback that runs every ``n`` full data-passes.

   :param Int call_on_0: keyword argument, default false. Unless set, the callback
          will **not** be run on epoch 0. Epoch 0 means no training has been performed
          yet. This is useful if you want to inspect the randomly initialized model
          that has not seen any data yet.

   :seealso: :func:`every_n_iter`.




.. function:: do_checkpoint(prefix; frequency=1, save_epoch_0=false)

   Create an :class:`AbstractEpochCallback` that save checkpoints of the model to disk.
   The checkpoints can be loaded back later on.

   :param AbstractString prefix: the prefix of the filenames to save the model. The model
          architecture will be saved to prefix-symbol.json, while the weights will be saved
          to prefix-0012.params, for example, for the 12-th epoch.
   :param Int frequency: keyword argument, default 1. The frequency (measured in epochs) to
          save checkpoints.
   :param Bool save_epoch_0: keyword argument, default false. Whether we should save a
          checkpoint for epoch 0 (model initialized but not seen any data yet).



