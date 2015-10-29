
Optimizers
==========

Common interfaces
-----------------




.. class:: AbstractOptimizer

   Base type for all optimizers.




.. class:: AbstractLearningRateScheduler

   Base type for all learning rate scheduler.




.. class:: AbstractMomentumScheduler

   Base type for all momentum scheduler.




.. class:: OptimizationState

   .. attribute:: batch_size

      The size of the mini-batch used in stochastic training.

   .. attribute:: curr_epoch

      The current epoch count. Epoch 0 means no training yet, during the first
      pass through the data, the epoch will be 1; during the second pass, the
      epoch count will be 1, and so on.

   .. attribute:: curr_batch

      The current mini-batch count. The batch count is reset during every epoch.
      The batch count 0 means the beginning of each epoch, with no mini-batch
      seen yet. During the first mini-batch, the mini-batch count will be 1.

   .. attribute:: curr_iter

      The current iteration count. One iteration corresponds to one mini-batch,
      but unlike the mini-batch count, the iteration count does **not** reset
      in each epoch. So it track the *total* number of mini-batches seen so far.




.. function:: get_learning_rate(scheduler, state)

   :param AbstractLearningRateScheduler scheduler: a learning rate scheduler.
   :param OptimizationState state: the current state about epoch, mini-batch and iteration count.
   :return: the current learning rate.




.. class:: LearningRate.Fixed

   Fixed learning rate scheduler always return the same learning rate.




.. class:: LearningRate.Exp

   :math:`\eta_t = \eta_0\gamma^t`. Here :math:`t` is the epoch count, or the iteration
   count if ``decay_on_iteration`` is set to true.




.. function:: get_momentum(scheduler, state)

   :param AbstractMomentumScheduler scheduler: the momentum scheduler.
   :param OptimizationState state: the state about current epoch, mini-batch and iteration count.
   :return: the current momentum.




.. class:: Momentum.Null

   The null momentum scheduler always returns 0 for momentum. It is also used to
   explicitly indicate momentum should not be used.




.. class:: Momentum.Fixed

  Fixed momentum scheduler always returns the same value.




.. function:: get_updater(optimizer)

   :param AbstractOptimizer optimizer: the underlying optimizer.

   A utility function to create an updater function, that uses its closure to
   store all the states needed for each weights.




Built-in optimizers
-------------------




.. class:: AbstractOptimizerOptions

   Base class for all optimizer options.




.. function:: normalized_gradient(opts, state, grad)

   :param AbstractOptimizerOptions opts: options for the optimizer, should contain the field
          ``grad_scale``, ``grad_clip`` and ``weight_decay``.
   :param OptimizationState state: the current optimization state.
   :param NDArray weight: the trainable weights.
   :param NDArray grad: the original gradient of the weights.

   Get the properly normalized gradient (re-scaled and clipped if necessary).



