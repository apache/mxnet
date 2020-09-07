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

# pylint: disable=fixme, too-many-arguments, too-many-locals, no-else-raise
# pylint: disable=too-many-public-methods, too-many-branches, too-many-lines
"""`BaseModule` defines an API for modules."""

import time
import logging
import warnings
import numpy as np

from .. import metric
from .. import ndarray

from ..context import cpu
from ..model import BatchEndParam
from ..initializer import Uniform
from ..io import DataDesc, DataIter, DataBatch
from ..base import _as_list


def _check_input_names(symbol, names, typename, throw):
    """Check that all input names are in symbol's arguments."""
    args = symbol.list_arguments()
    for name in names:
        if name in args:
            continue
        candidates = [arg for arg in args if
                      not arg.endswith('_weight') and
                      not arg.endswith('_bias') and
                      not arg.endswith('_gamma') and
                      not arg.endswith('_beta')]
        msg = "\033[91mYou created Module with Module(..., %s_names=%s) but " \
              "input with name '%s' is not found in symbol.list_arguments(). " \
              "Did you mean one of:\n\t%s\033[0m"%(
                  typename, str(names), name, '\n\t'.join(candidates))
        if throw:
            raise ValueError(msg)
        else:
            warnings.warn(msg)


def _check_names_match(data_names, data_shapes, name, throw):
    """Check that input names matches input data descriptors."""
    actual = [x[0] for x in data_shapes]
    if sorted(data_names) != sorted(actual):
        msg = "Data provided by %s_shapes don't match names specified by %s_names (%s vs. %s)"%(
            name, name, str(data_shapes), str(data_names))
        if throw:
            raise ValueError(msg)
        else:
            warnings.warn(msg)


def _parse_data_desc(data_names, label_names, data_shapes, label_shapes):
    """parse data_attrs into DataDesc format and check that names match"""
    data_shapes = [x if isinstance(x, DataDesc) else DataDesc(*x) for x in data_shapes]
    _check_names_match(data_names, data_shapes, 'data', True)
    if label_shapes is not None:
        label_shapes = [x if isinstance(x, DataDesc) else DataDesc(*x) for x in label_shapes]
        _check_names_match(label_names, label_shapes, 'label', False)
    else:
        _check_names_match(label_names, [], 'label', False)
    return data_shapes, label_shapes


class BaseModule(object):
    """The base class of a module.

    A module represents a computation component. One can think of module as a computation machine.
    A module can execute forward and backward passes and update parameters in a model.
    We aim to make the APIs easy to use, especially in the case when we need to use the imperative
    API to work with multiple modules (e.g. stochastic depth network).

    A module has several states:

    - Initial state: Memory is not allocated yet, so the module is not ready for computation yet.
    - Binded: Shapes for inputs, outputs, and parameters are all known, memory has been allocated,
      and the module is ready for computation.
    - Parameters are initialized: For modules with parameters, doing computation before
      initializing the parameters might result in undefined outputs.
    - Optimizer is installed: An optimizer can be installed to a module. After this, the parameters
      of the module can be updated according to the optimizer after gradients are computed
      (forward-backward).

    In order for a module to interact with others, it must be able to report the
    following information in its initial state (before binding):

    - `data_names`: list of type string indicating the names of the required input data.
    - `output_names`: list of type string indicating the names of the required outputs.

    After binding, a module should be able to report the following richer information:

    - state information
        - `binded`: `bool`, indicates whether the memory buffers needed for computation
          have been allocated.
        - `for_training`: whether the module is bound for training.
        - `params_initialized`: `bool`, indicates whether the parameters of this module
          have been initialized.
        - `optimizer_initialized`: `bool`, indicates whether an optimizer is defined
          and initialized.
        - `inputs_need_grad`: `bool`, indicates whether gradients with respect to the
          input data are needed. Might be useful when implementing composition of modules.

    - input/output information
        - `data_shapes`: a list of `(name, shape)`. In theory, since the memory is allocated,
          we could directly provide the data arrays. But in the case of data parallelism,
          the data arrays might not be of the same shape as viewed from the external world.
        - `label_shapes`: a list of `(name, shape)`. This might be `[]` if the module does
          not need labels (e.g. it does not contains a loss function at the top), or a module
          is not bound for training.
        - `output_shapes`: a list of `(name, shape)` for outputs of the module.

    - parameters (for modules with parameters)
        - `get_params()`: return a tuple `(arg_params, aux_params)`. Each of those
          is a dictionary of name to ``NDArray`` mapping. Those `NDArray` always lives on
          CPU. The actual parameters used for computing might live on other devices (GPUs),
          this function will retrieve (a copy of) the latest parameters.
        - ``set_params(arg_params, aux_params)``: assign parameters to the devices
          doing the computation.
        - ``init_params(...)``: a more flexible interface to assign or initialize the parameters.

    - setup
        - `bind()`: prepare environment for computation.
        - `init_optimizer()`: install optimizer for parameter updating.
        - `prepare()`: prepare the module based on the current data batch.

    - computation
        - `forward(data_batch)`: forward operation.
        - `backward(out_grads=None)`: backward operation.
        - `update()`: update parameters according to installed optimizer.
        - `get_outputs()`: get outputs of the previous forward operation.
        - `get_input_grads()`: get the gradients with respect to the inputs computed
          in the previous backward operation.
        - `update_metric(metric, labels, pre_sliced=False)`: update performance metric
          for the previous forward
          computed results.

    - other properties (mostly for backward compatibility)
        - `symbol`: the underlying symbolic graph for this module (if any)
          This property is not necessarily constant. For example, for `BucketingModule`,
          this property is simply the *current* symbol being used. For other modules,
          this value might not be well defined.

    When those intermediate-level API are implemented properly, the following
    high-level API will be automatically available for a module:

    - `fit`: train the module parameters on a data set.
    - `predict`: run prediction on a data set and collect outputs.
    - `score`: run prediction on a data set and evaluate performance.

    Examples
    --------
    >>> # An example of creating a mxnet module.
    >>> import mxnet as mx
    >>> data = mx.symbol.Variable('data')
    >>> fc1  = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
    >>> act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
    >>> fc2  = mx.symbol.FullyConnected(act1, name = 'fc2', num_hidden = 64)
    >>> act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
    >>> fc3  = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=10)
    >>> out  = mx.symbol.SoftmaxOutput(fc3, name = 'softmax')
    >>> mod = mx.mod.Module(out)
    """
    def __init__(self, logger=logging):
        self.logger = logger
        self.binded = False
        self.for_training = False
        self.inputs_need_grad = False
        self.params_initialized = False
        self.optimizer_initialized = False
        self._symbol = None
        self._total_exec_bytes = 0

    ################################################################################
    # High Level API
    ################################################################################
    def forward_backward(self, data_batch):
        """A convenient function that calls both ``forward`` and ``backward``."""
        self.forward(data_batch, is_train=True)
        self.backward()

    def score(self, eval_data, eval_metric, num_batch=None, batch_end_callback=None,
              score_end_callback=None,
              reset=True, epoch=0, sparse_row_id_fn=None):
        """Runs prediction on ``eval_data`` and evaluates the performance according to
        the given ``eval_metric``.

        Checkout `Module Tutorial <https://mxnet.apache.org/api/python/tutorials/packages/module/index.html>`_
        to see an end-to-end use-case.

        Parameters
        ----------
        eval_data : DataIter
            Evaluation data to run prediction on.
        eval_metric : EvalMetric or list of EvalMetrics
            Evaluation metric to use.
        num_batch : int
            Number of batches to run. Defaults to ``None``, indicating run until the `DataIter`
            finishes.
        batch_end_callback : function
            Could also be a list of functions.
        reset : bool
            Defaults to ``True``. Indicates whether we should reset `eval_data` before starting
            evaluating.
        epoch : int
            Defaults to 0. For compatibility, this will be passed to callbacks (if any).
            During training, this will correspond to the training epoch number.
        sparse_row_id_fn : A callback function
            The function  takes `data_batch` as an input and returns a dict of
            str -> NDArray. The resulting dict is used for pulling row_sparse
            parameters from the kvstore, where the str key is the name of the param,
            and the value is the row id of the param to pull.

        Examples
        --------
        >>> # An example of using score for prediction.
        >>> # Evaluate accuracy on val_dataiter
        >>> metric = mx.metric.Accuracy()
        >>> mod.score(val_dataiter, metric)
        >>> mod.score(val_dataiter, ['mse', 'acc'])
        """
        assert self.binded and self.params_initialized

        if reset:
            eval_data.reset()

        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        eval_metric.reset()
        actual_num_batch = 0

        for nbatch, eval_batch in enumerate(eval_data):
            if num_batch is not None and nbatch == num_batch:
                break
            self.prepare(eval_batch, sparse_row_id_fn=sparse_row_id_fn)
            self.forward(eval_batch, is_train=False)
            if isinstance(eval_batch, list):
                self.update_metric(eval_metric, [eb.label for eb in eval_batch], pre_sliced=True)
            else:
                self.update_metric(eval_metric, eval_batch.label)

            if batch_end_callback is not None:
                batch_end_params = BatchEndParam(epoch=epoch,
                                                 nbatch=nbatch,
                                                 eval_metric=eval_metric,
                                                 locals=locals())
                for callback in _as_list(batch_end_callback):
                    callback(batch_end_params)
            actual_num_batch += 1

        if score_end_callback:
            params = BatchEndParam(epoch=epoch,
                                   nbatch=actual_num_batch,
                                   eval_metric=eval_metric,
                                   locals=locals())
            for callback in _as_list(score_end_callback):
                callback(params)

        return eval_metric.get_name_value()

    def iter_predict(self, eval_data, num_batch=None, reset=True, sparse_row_id_fn=None):
        """Iterates over predictions.

        Examples
        --------
        >>> for pred, i_batch, batch in module.iter_predict(eval_data):
        ...     # pred is a list of outputs from the module
        ...     # i_batch is a integer
        ...     # batch is the data batch from the data iterator

        Parameters
        ----------
        eval_data : DataIter
            Evaluation data to run prediction on.
        num_batch : int
            Default is ``None``, indicating running all the batches in the data iterator.
        reset : bool
            Default is ``True``, indicating whether we should reset the data iter before start
            doing prediction.
        sparse_row_id_fn : A callback function
            The function  takes `data_batch` as an input and returns a dict of
            str -> NDArray. The resulting dict is used for pulling row_sparse
            parameters from the kvstore, where the str key is the name of the param,
            and the value is the row id of the param to pull.
        """
        assert self.binded and self.params_initialized

        if reset:
            eval_data.reset()

        for nbatch, eval_batch in enumerate(eval_data):
            if num_batch is not None and nbatch == num_batch:
                break
            self.prepare(eval_batch, sparse_row_id_fn=sparse_row_id_fn)
            self.forward(eval_batch, is_train=False)
            pad = eval_batch.pad
            outputs = [out[0:out.shape[0]-pad] for out in self.get_outputs()]

            yield (outputs, nbatch, eval_batch)

    def predict(self, eval_data, num_batch=None, merge_batches=True, reset=True,
                always_output_list=False, sparse_row_id_fn=None):
        """Runs prediction and collects the outputs.

        When `merge_batches` is ``True`` (by default), the return value will be a list
        ``[out1, out2, out3]``, where each element is formed by concatenating the outputs for
        all the mini-batches. When `always_output_list` is ``False`` (as by default),
        then in the case of a single output, `out1` is returned instead of ``[out1]``.

        When `merge_batches` is ``False``, the return value will be a nested list like
        ``[[out1_batch1, out2_batch1], [out1_batch2], ...]``. This mode is useful because
        in some cases (e.g. bucketing), the module does not necessarily produce the same
        number of outputs.

        The objects in the results have type `NDArray`. If you need to work with a numpy array,
        just call ``.asnumpy()`` on each `NDArray`.

        Parameters
        ----------
        eval_data : DataIter or NDArray or numpy array
            Evaluation data to run prediction on.
        num_batch : int
            Defaults to ``None``, indicates running all the batches in the data iterator.
        merge_batches : bool
            Defaults to ``True``, see above for return values.
        reset : bool
            Defaults to ``True``, indicates whether we should reset the data iter before
            doing prediction.
        always_output_list : bool
            Defaults to ``False``, see above for return values.
        sparse_row_id_fn : A callback function
            The function  takes `data_batch` as an input and returns a dict of
            str -> NDArray. The resulting dict is used for pulling row_sparse
            parameters from the kvstore, where the str key is the name of the param,
            and the value is the row id of the param to pull.

        Returns
        -------
        list of NDArray or list of list of NDArray
            Prediction results.

        Examples
        --------
        >>> # An example of using `predict` for prediction.
        >>> # Predict on the first 10 batches of val_dataiter
        >>> mod.predict(eval_data=val_dataiter, num_batch=10)
        """
        assert self.binded and self.params_initialized

        if isinstance(eval_data, (ndarray.NDArray, np.ndarray)):
            if isinstance(eval_data, np.ndarray):
                eval_data = ndarray.array(eval_data)
            self.forward(DataBatch([eval_data]))
            return self.get_outputs()[0]

        if not isinstance(eval_data, DataIter):
            raise ValueError('eval_data must be of type NDArray or DataIter')

        if reset:
            eval_data.reset()

        output_list = []

        for nbatch, eval_batch in enumerate(eval_data):
            if num_batch is not None and nbatch == num_batch:
                break
            self.prepare(eval_batch, sparse_row_id_fn=sparse_row_id_fn)
            self.forward(eval_batch, is_train=False)
            pad = eval_batch.pad
            outputs = [out[0:out.shape[0]-pad].copy() for out in self.get_outputs()]

            output_list.append(outputs)

        if len(output_list) == 0:
            return output_list

        if merge_batches:
            num_outputs = len(output_list[0])
            for out in output_list:
                assert len(out) == num_outputs, \
                       'Cannot merge batches, as num of outputs is not the same ' + \
                       'in mini-batches. Maybe bucketing is used?'
            output_list2 = [ndarray.concatenate([out[i] for out in output_list])
                            for i in range(num_outputs)]

            if num_outputs == 1 and not always_output_list:
                return output_list2[0]
            return output_list2

        return output_list

    def fit(self, train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_end_callback=None,
            eval_batch_end_callback=None, initializer=Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None, monitor=None, sparse_row_id_fn=None):
        """Trains the module parameters.

        Checkout `Module Tutorial <https://mxnet.apache.org/api/python/tutorials/packages/module/index.html>`_
        to see an end-to-end use-case.

        Parameters
        ----------
        train_data : DataIter
            Train DataIter.
        eval_data : DataIter
            If not ``None``, will be used as validation set and the performance
            after each epoch will be evaluated.
        eval_metric : str or EvalMetric
            Defaults to 'accuracy'. The performance measure used to display during training.
            Other possible predefined metrics are:
            'ce' (CrossEntropy), 'f1', 'mae', 'mse', 'rmse', 'top_k_accuracy'.
        epoch_end_callback : function or list of functions
            Each callback will be called with the current `epoch`, `symbol`, `arg_params`
            and `aux_params`.
        batch_end_callback : function or list of function
            Each callback will be called with a `BatchEndParam`.
        kvstore : str or KVStore
            Defaults to 'local'.
        optimizer : str or Optimizer
            Defaults to 'sgd'.
        optimizer_params : dict
            Defaults to ``(('learning_rate', 0.01),)``. The parameters for
            the optimizer constructor.
            The default value is not a dict, just to avoid pylint warning on dangerous
            default values.
        eval_end_callback : function or list of function
            These will be called at the end of each full evaluation, with the metrics over
            the entire evaluation set.
        eval_batch_end_callback : function or list of function
            These will be called at the end of each mini-batch during evaluation.
        initializer : Initializer
            The initializer is called to initialize the module parameters when they are
            not already initialized.
        arg_params : dict
            Defaults to ``None``, if not ``None``, should be existing parameters from a trained
            model or loaded from a checkpoint (previously saved model). In this case,
            the value here will be used to initialize the module parameters, unless they
            are already initialized by the user via a call to `init_params` or `fit`.
            `arg_params` has a higher priority than `initializer`.
        aux_params : dict
            Defaults to ``None``. Similar to `arg_params`, except for auxiliary states.
        allow_missing : bool
            Defaults to ``False``. Indicates whether to allow missing parameters when `arg_params`
            and `aux_params` are not ``None``. If this is ``True``, then the missing parameters
            will be initialized via the `initializer`.
        force_rebind : bool
            Defaults to ``False``. Whether to force rebinding the executors if already bound.
        force_init : bool
            Defaults to ``False``. Indicates whether to force initialization even if the
            parameters are already initialized.
        begin_epoch : int
            Defaults to 0. Indicates the starting epoch. Usually, if resumed from a
            checkpoint saved at a previous training phase at epoch N, then this value should be
            N+1.
        num_epoch : int
            Number of epochs for training.
        sparse_row_id_fn : A callback function
            The function  takes `data_batch` as an input and returns a dict of
            str -> NDArray. The resulting dict is used for pulling row_sparse
            parameters from the kvstore, where the str key is the name of the param,
            and the value is the row id of the param to pull.

        Examples
        --------
        >>> # An example of using fit for training.
        >>> # Assume training dataIter and validation dataIter are ready
        >>> # Assume loading a previously checkpointed model
        >>> sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 3)
        >>> mod.fit(train_data=train_dataiter, eval_data=val_dataiter, optimizer='sgd',
        ...     optimizer_params={'learning_rate':0.01, 'momentum': 0.9},
        ...     arg_params=arg_params, aux_params=aux_params,
        ...     eval_metric='acc', num_epoch=10, begin_epoch=3)
        """
        assert num_epoch is not None, 'please specify number of epochs'

        self.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
                  for_training=True, force_rebind=force_rebind)
        if monitor is not None:
            self.install_monitor(monitor)
        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)
        self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)

        if validation_metric is None:
            validation_metric = eval_metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        ################################################################################
        # training loop
        ################################################################################
        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            eval_metric.reset()
            nbatch = 0
            data_iter = iter(train_data)
            end_of_batch = False
            next_data_batch = next(data_iter)
            while not end_of_batch:
                data_batch = next_data_batch
                if monitor is not None:
                    monitor.tic()
                self.forward_backward(data_batch)
                self.update()

                if isinstance(data_batch, list):
                    self.update_metric(eval_metric,
                                       [db.label for db in data_batch],
                                       pre_sliced=True)
                else:
                    self.update_metric(eval_metric, data_batch.label)

                try:
                    # pre fetch next batch
                    next_data_batch = next(data_iter)
                    self.prepare(next_data_batch, sparse_row_id_fn=sparse_row_id_fn)
                except StopIteration:
                    end_of_batch = True

                if monitor is not None:
                    monitor.toc_print()

                if end_of_batch:
                    eval_name_vals = eval_metric.get_global_name_value()

                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    for callback in _as_list(batch_end_callback):
                        callback(batch_end_params)
                nbatch += 1

            # one epoch of training is finished
            for name, val in eval_name_vals:
                self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            # sync aux params across devices
            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)

            if epoch_end_callback is not None:
                for callback in _as_list(epoch_end_callback):
                    callback(epoch, self.symbol, arg_params, aux_params)

            #----------------------------------------
            # evaluation on validation set
            if eval_data is not None:
                res = self.score(eval_data, validation_metric,
                                 score_end_callback=eval_end_callback,
                                 batch_end_callback=eval_batch_end_callback, epoch=epoch)
                #TODO: pull this into default
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

            # end of 1 epoch, reset the data-iter for another epoch
            train_data.reset()

    ################################################################################
    # Symbol information
    ################################################################################
    @property
    def data_names(self):
        """A list of names for data required by this module."""
        raise NotImplementedError()

    @property
    def output_names(self):
        """A list of names for the outputs of this module."""
        raise NotImplementedError()

    ################################################################################
    # Input/Output information
    ################################################################################
    @property
    def data_shapes(self):
        """A list of (name, shape) pairs specifying the data inputs to this module."""
        raise NotImplementedError()

    @property
    def label_shapes(self):
        """A list of (name, shape) pairs specifying the label inputs to this module.
        If this module does not accept labels -- either it is a module without loss
        function, or it is not bound for training, then this should return an empty
        list ``[]``.
        """
        raise NotImplementedError()

    @property
    def output_shapes(self):
        """A list of (name, shape) pairs specifying the outputs of this module."""
        raise NotImplementedError()

    ################################################################################
    # Parameters of a module
    ################################################################################
    def get_params(self):
        """Gets parameters, those are potentially copies of the actual parameters used
        to do computation on the device.

        Returns
        -------
        ``(arg_params, aux_params)``
            A pair of dictionaries each mapping parameter names to NDArray values.

        Examples
        --------
        >>> # An example of getting module parameters.
        >>> print mod.get_params()
        ({'fc2_weight': <NDArray 64x128 @cpu(0)>, 'fc1_weight': <NDArray 128x100 @cpu(0)>,
        'fc3_bias': <NDArray 10 @cpu(0)>, 'fc3_weight': <NDArray 10x64 @cpu(0)>,
        'fc2_bias': <NDArray 64 @cpu(0)>, 'fc1_bias': <NDArray 128 @cpu(0)>}, {})
        """
        raise NotImplementedError()

    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False, allow_extra=False):
        """Initializes the parameters and auxiliary states.

        Parameters
        ----------
        initializer : Initializer
            Called to initialize parameters if needed.
        arg_params : dict
            If not ``None``, should be a dictionary of existing `arg_params`. Initialization
            will be copied from that.
        aux_params : dict
            If not ``None``, should be a dictionary of existing `aux_params`. Initialization
            will be copied from that.
        allow_missing : bool
            If ``True``, params could contain missing values, and the initializer will be
            called to fill those missing params.
        force_init : bool
            If ``True``, `force_init` will force re-initialize even if already initialized.
        allow_extra : boolean, optional
            Whether allow extra parameters that are not needed by symbol.
            If this is True, no error will be thrown when arg_params or aux_params
            contain extra parameters that is not needed by the executor.

        Examples
        --------
        >>> # An example of initializing module parameters.
        >>> mod.init_params()
        """
        raise NotImplementedError()

    def set_params(self, arg_params, aux_params, allow_missing=False, force_init=True,
                   allow_extra=False):
        """Assigns parameter and aux state values.

        Parameters
        ----------
        arg_params : dict
            Dictionary of name to value (`NDArray`) mapping.
        aux_params : dict
            Dictionary of name to value (`NDArray`) mapping.
        allow_missing : bool
            If ``True``, params could contain missing values, and the initializer will be
            called to fill those missing params.
        force_init : bool
            If ``True``, will force re-initialize even if already initialized.
        allow_extra : boolean, optional
            Whether allow extra parameters that are not needed by symbol.
            If this is True, no error will be thrown when arg_params or aux_params
            contain extra parameters that is not needed by the executor.

        Examples
        --------
        >>> # An example of setting module parameters.
        >>> sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, n_epoch_load)
        >>> mod.set_params(arg_params=arg_params, aux_params=aux_params)
        """
        self.init_params(initializer=None, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init,
                         allow_extra=allow_extra)

    def save_params(self, fname):
        """Saves model parameters to file.

        Parameters
        ----------
        fname : str
            Path to output param file.

        Examples
        --------
        >>> # An example of saving module parameters.
        >>> mod.save_params('myfile')
        """
        arg_params, aux_params = self.get_params()
        save_dict = {('arg:%s' % k) : v.as_in_context(cpu()) for k, v in arg_params.items()}
        save_dict.update({('aux:%s' % k) : v.as_in_context(cpu()) for k, v in aux_params.items()})
        ndarray.save(fname, save_dict)

    def load_params(self, fname):
        """Loads model parameters from file.

        Parameters
        ----------
        fname : str
            Path to input param file.

        Examples
        --------
        >>> # An example of loading module parameters.
        >>> mod.load_params('myfile')
        """
        save_dict = ndarray.load(fname)
        arg_params = {}
        aux_params = {}
        for k, value in save_dict.items():
            arg_type, name = k.split(':', 1)
            if arg_type == 'arg':
                arg_params[name] = value
            elif arg_type == 'aux':
                aux_params[name] = value
            else:
                raise ValueError("Invalid param file " + fname)
        self.set_params(arg_params, aux_params)

    def get_states(self, merge_multi_context=True):
        """Gets states from all devices

        If `merge_multi_context` is ``True``, returns output of form ``[out1, out2]``.
        Otherwise, it returns output of the form
        ``[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]``.
        All output elements are `NDArray`.

        Parameters
        ----------
        merge_multi_context : bool
            Defaults to ``True``. In the case when data-parallelism is used, the states
            will be collected from multiple devices. A ``True`` value indicates that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        A list of ``NDArray`` or a list of list of ``NDArray``.
        """
        assert self.binded and self.params_initialized
        assert not merge_multi_context
        return []

    def set_states(self, states=None, value=None):
        """Sets value for states. Only one of states & value can be specified.

        Parameters
        ----------
        states : list of list of NDArray
            Source states arrays formatted like
            ``[[state1_dev1, state1_dev2], [state2_dev1, state2_dev2]]``.
        value : number
            A single scalar value for all state arrays.
        """
        assert self.binded and self.params_initialized
        assert not states and not value

    def install_monitor(self, mon):
        """Installs monitor on all executors."""
        raise NotImplementedError()

    ################################################################################
    # Computations
    ################################################################################
    # pylint: disable=unused-argument
    def prepare(self, data_batch, sparse_row_id_fn=None):
        '''Prepares the module for processing a data batch.

        Usually involves switching bucket and reshaping.
        For modules that contain `row_sparse` parameters in KVStore,
        it prepares the `row_sparse` parameters based on the sparse_row_id_fn.

        When KVStore is used to update parameters for multi-device or multi-machine training,
        a copy of the parameters are stored in KVStore. Note that for `row_sparse` parameters,
        the `update()` updates the copy of parameters in KVStore, but doesn't broadcast
        the updated parameters to all devices / machines. The `prepare` function is used to
        broadcast `row_sparse` parameters with the next batch of data.

        Parameters
        ----------
        data_batch : DataBatch
            The current batch of data for forward computation.

        sparse_row_id_fn : A callback function
            The function  takes `data_batch` as an input and returns a dict of
            str -> NDArray. The resulting dict is used for pulling row_sparse
            parameters from the kvstore, where the str key is the name of the param,
            and the value is the row id of the param to pull.
        '''
        if sparse_row_id_fn is not None:
            warnings.warn(UserWarning("sparse_row_id_fn is not invoked for BaseModule."))
    # pylint: enable=unused-argument

    def forward(self, data_batch, is_train=None):
        """Forward computation. It supports data batches with different shapes, such as
        different batch sizes or different image sizes.
        If reshaping of data batch relates to modification of symbol or module, such as
        changing image layout ordering or switching from training to predicting, module
        rebinding is required.

        Parameters
        ----------
        data_batch : DataBatch
            Could be anything with similar API implemented.
        is_train : bool
            Default is ``None``, which means `is_train` takes the value of ``self.for_training``.

        Examples
        --------
        >>> import mxnet as mx
        >>> from collections import namedtuple
        >>> Batch = namedtuple('Batch', ['data'])
        >>> data = mx.sym.Variable('data')
        >>> out = data * 2
        >>> mod = mx.mod.Module(symbol=out, label_names=None)
        >>> mod.bind(data_shapes=[('data', (1, 10))])
        >>> mod.init_params()
        >>> data1 = [mx.nd.ones((1, 10))]
        >>> mod.forward(Batch(data1))
        >>> print mod.get_outputs()[0].asnumpy()
        [[ 2.  2.  2.  2.  2.  2.  2.  2.  2.  2.]]
        >>> # Forward with data batch of different shape
        >>> data2 = [mx.nd.ones((3, 5))]
        >>> mod.forward(Batch(data2))
        >>> print mod.get_outputs()[0].asnumpy()
        [[ 2.  2.  2.  2.  2.]
         [ 2.  2.  2.  2.  2.]
         [ 2.  2.  2.  2.  2.]]
        """
        raise NotImplementedError()

    def backward(self, out_grads=None):
        """Backward computation.

        Parameters
        ----------
        out_grads : NDArray or list of NDArray, optional
            Gradient on the outputs to be propagated back.
            This parameter is only needed when bind is called
            on outputs that are not a loss function.

        Examples
        --------
        >>> # An example of backward computation.
        >>> mod.backward()
        >>> print mod.get_input_grads()[0].asnumpy()
        [[[  1.10182791e-05   5.12257748e-06   4.01927764e-06   8.32566820e-06
            -1.59775993e-06   7.24269375e-06   7.28067835e-06  -1.65902311e-05
             5.46342608e-06   8.44196393e-07]
             ...]]
        """
        raise NotImplementedError()

    def get_outputs(self, merge_multi_context=True):
        """Gets outputs of the previous forward computation.

        If `merge_multi_context` is ``True``, it is like ``[out1, out2]``. Otherwise,
        it returns out put of form ``[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]``.
        All the output elements have type `NDArray`. When `merge_multi_context` is ``False``,
        those `NDArray` instances might live on different devices.

        Parameters
        ----------
        merge_multi_context : bool
            Defaults to ``True``. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A ``True`` value indicates that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        list of `NDArray` or list of list of `NDArray`.
            Output

        Examples
        --------
        >>> # An example of getting forward output.
        >>> print mod.get_outputs()[0].asnumpy()
        [[ 0.09999977  0.10000153  0.10000716  0.10000195  0.09999853  0.09999743
           0.10000272  0.10000113  0.09999088  0.09999888]]
        """
        raise NotImplementedError()

    def get_input_grads(self, merge_multi_context=True):
        """Gets the gradients to the inputs, computed in the previous backward computation.

        If `merge_multi_context` is ``True``, it is like ``[grad1, grad2]``. Otherwise, it
        is like ``[[grad1_dev1, grad1_dev2], [grad2_dev1, grad2_dev2]]``. All the output
        elements have type `NDArray`. When `merge_multi_context` is ``False``, those `NDArray`
        instances might live on different devices.

        Parameters
        ----------
        merge_multi_context : bool
            Defaults to ``True``. In the case when data-parallelism is used, the gradients
            will be collected from multiple devices. A ``True`` value indicates that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        list of NDArray or list of list of NDArray
              Input gradients.

        Examples
        --------
        >>> # An example of getting input gradients.
        >>> print mod.get_input_grads()[0].asnumpy()
        [[[  1.10182791e-05   5.12257748e-06   4.01927764e-06   8.32566820e-06
            -1.59775993e-06   7.24269375e-06   7.28067835e-06  -1.65902311e-05
            5.46342608e-06   8.44196393e-07]
            ...]]
        """
        raise NotImplementedError()

    def update(self):
        """Updates parameters according to the installed optimizer and the gradients computed
        in the previous forward-backward batch.

        When KVStore is used to update parameters for multi-device or multi-machine training,
        a copy of the parameters are stored in KVStore. Note that for `row_sparse` parameters,
        this function does update the copy of parameters in KVStore, but doesn't broadcast the
        updated parameters to all devices / machines. Please call `prepare` to broadcast
        `row_sparse` parameters with the next batch of data.

        Examples
        --------
        >>> # An example of updating module parameters.
        >>> mod.init_optimizer(kvstore='local', optimizer='sgd',
        ...     optimizer_params=(('learning_rate', 0.01), ))
        >>> mod.backward()
        >>> mod.update()
        >>> print mod.get_params()[0]['fc3_weight'].asnumpy()
        [[  5.86930104e-03   5.28078526e-03  -8.88729654e-03  -1.08308345e-03
            6.13054074e-03   4.27560415e-03   1.53817423e-03   4.62131854e-03
            4.69872449e-03  -2.42400169e-03   9.94111411e-04   1.12386420e-03
            ...]]
        """
        raise NotImplementedError()

    def update_metric(self, eval_metric, labels, pre_sliced=False):
        """Evaluates and accumulates evaluation metric on outputs of the last forward
        computation.

        Parameters
        ----------
        eval_metric : EvalMetric
            Evaluation metric to use.
        labels : list of NDArray if `pre_sliced` parameter is set to `False`,
            list of lists of NDArray otherwise. Typically `data_batch.label`.
        pre_sliced: bool
            Whether the labels are already sliced per device (default: False).

        Examples
        --------
        >>> # An example of updating evaluation metric.
        >>> mod.forward(data_batch)
        >>> mod.update_metric(metric, data_batch.label)
        """
        raise NotImplementedError()

    ################################################################################
    # module setup
    ################################################################################
    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None,
             grad_req='write'):
        """Binds the symbols to construct executors. This is necessary before one
        can perform computation with the module.

        Parameters
        ----------
        data_shapes : list of (str, tuple) or DataDesc objects
            Typically is ``data_iter.provide_data``. Can also be a list of
            (data name, data shape).
        label_shapes : list of (str, tuple) or DataDesc objects
            Typically is ``data_iter.provide_label``. Can also be a list of
            (label name, label shape).
        for_training : bool
            Default is ``True``. Whether the executors should be bind for training.
        inputs_need_grad : bool
            Default is ``False``. Whether the gradients to the input data need to be computed.
            Typically this is not needed. But this might be needed when implementing composition
            of modules.
        force_rebind : bool
            Default is ``False``. This function does nothing if the executors are already
            bound. But with this ``True``, the executors will be forced to rebind.
        shared_module : Module
            Default is ``None``. This is used in bucketing. When not ``None``, the shared module
            essentially corresponds to a different bucket -- a module with different symbol
            but with the same sets of parameters (e.g. unrolled RNNs with different lengths).
        grad_req : str, list of str, dict of str to str
            Requirement for gradient accumulation. Can be 'write', 'add', or 'null'
            (default to 'write').
            Can be specified globally (str) or for each argument (list, dict).

        Examples
        --------
        >>> # An example of binding symbols.
        >>> mod.bind(data_shapes=[('data', (1, 10, 10))])
        >>> # Assume train_iter is already created.
        >>> mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
        """
        raise NotImplementedError()

    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),), force_init=False):
        """Installs and initializes optimizers, as well as initialize kvstore for
           distributed training

        Parameters
        ----------
        kvstore : str or KVStore
            Defaults to `'local'`.
        optimizer : str or Optimizer
            Defaults to `'sgd'`.
        optimizer_params : dict
            Defaults to ``(('learning_rate', 0.01),)``. The default value is not a dictionary,
            just to avoid pylint warning of dangerous default values.
        force_init : bool
            Defaults to ``False``, indicates whether to force re-initializing an optimizer
            if it is already installed.

        Examples
        --------
        >>> # An example of initializing optimizer.
        >>> mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.005),))
        """
        raise NotImplementedError()

    ################################################################################
    # misc
    ################################################################################
    @property
    def symbol(self):
        """Gets the symbol associated with this module.

        Except for `Module`, for other types of modules (e.g. `BucketingModule`), this
        property might not be a constant throughout its life time. Some modules might
        not even be associated with any symbols.
        """
        return self._symbol
