# pylint: disable=fixme, invalid-name, too-many-arguments, too-many-locals, too-many-lines
# pylint: disable=too-many-branches, too-many-statements
"""MXNet model module"""
from __future__ import absolute_import, print_function

import time
import logging
import warnings
from collections import namedtuple
import numpy as np

from . import io
from . import nd
from . import symbol as sym
from . import optimizer as opt
from . import metric
from . import kvstore as kvs
from .context import Context, cpu
from .initializer import Uniform
from .optimizer import get_updater
from .executor_manager import DataParallelExecutorManager, _check_arguments, _load_data
from .io import DataDesc
from .base import mx_real_t

BASE_ESTIMATOR = object

try:
    from sklearn.base import BaseEstimator
    BASE_ESTIMATOR = BaseEstimator
except ImportError:
    SKLEARN_INSTALLED = False

# Parameter to pass to batch_end_callback
BatchEndParam = namedtuple('BatchEndParams',
                           ['epoch',
                            'nbatch',
                            'eval_metric',
                            'locals'])

def _create_kvstore(kvstore, num_device, arg_params):
    """Create kvstore
    This function select and create a proper kvstore if given the kvstore type.

    Parameters
    ----------
    kvstore : KVStore or str
        The kvstore.
    num_device : int
        The number of devices
    arg_params : dict of str to `NDArray`.
        Model parameter, dict of name to `NDArray` of net's weights.
    """
    update_on_kvstore = True
    if kvstore is None:
        kv = None
    elif isinstance(kvstore, kvs.KVStore):
        kv = kvstore
    elif isinstance(kvstore, str):
        # create kvstore using the string type
        if num_device is 1 and 'dist' not in kvstore:
            # no need to use kv for single device and single machine
            kv = None
        else:
            kv = kvs.create(kvstore)
            if kvstore == 'local':
            # automatically select a proper local
                max_size = max(np.prod(param.shape) for param in
                               arg_params.values())
                if max_size > 1024 * 1024 * 16:
                    update_on_kvstore = False
    else:
        raise TypeError('kvstore must be KVStore, str or None')

    if kv is None:
        update_on_kvstore = False

    return (kv, update_on_kvstore)

def _initialize_kvstore(kvstore, param_arrays, arg_params, param_names,
                        update_on_kvstore):
    """Initialize kvstore"""
    for idx, param_on_devs in enumerate(param_arrays):
        name = param_names[idx]
        kvstore.init(name, arg_params[name])

        if update_on_kvstore:
            kvstore.pull(name, param_on_devs, priority=-idx)

def _update_params_on_kvstore(param_arrays, grad_arrays, kvstore, param_names):
    """Perform update of param_arrays from grad_arrays on kvstore."""
    for index, pair in enumerate(zip(param_arrays, grad_arrays)):
        arg_list, grad_list = pair
        if grad_list[0] is None:
            continue
        name = param_names[index]
        # push gradient, priority is negative index
        kvstore.push(name, grad_list, priority=-index)
        # pull back the weights
        kvstore.pull(name, arg_list, priority=-index)

def _update_params(param_arrays, grad_arrays, updater, num_device,
                   kvstore=None, param_names=None):
    """Perform update of param_arrays from grad_arrays not on kvstore."""
    for index, pair in enumerate(zip(param_arrays, grad_arrays)):
        arg_list, grad_list = pair
        if grad_list[0] is None:
            continue
        if kvstore:
            name = param_names[index]
            # push gradient, priority is negative index
            kvstore.push(name, grad_list, priority=-index)
            # pull back the sum gradients, to the same locations.
            kvstore.pull(name, grad_list, priority=-index)
        for k, p in enumerate(zip(arg_list, grad_list)):
            # faked an index here, to make optimizer create diff
            # state for the same index but on diff devs, TODO(mli)
            # use a better solution latter
            w, g = p
            updater(index*num_device+k, g, w)


def _multiple_callbacks(callbacks, *args, **kwargs):
    """Sends args and kwargs to any configured callbacks.
    This handles the cases where the 'callbacks' variable
    is ``None``, a single function, or a list.
    """
    if isinstance(callbacks, list):
        for cb in callbacks:
            cb(*args, **kwargs)
        return
    if callbacks:
        callbacks(*args, **kwargs)


def _train_multi_device(symbol, ctx, arg_names, param_names, aux_names,
                        arg_params, aux_params,
                        begin_epoch, end_epoch, epoch_size, optimizer,
                        kvstore, update_on_kvstore,
                        train_data, eval_data=None, eval_metric=None,
                        epoch_end_callback=None, batch_end_callback=None,
                        logger=None, work_load_list=None, monitor=None,
                        eval_end_callback=None,
                        eval_batch_end_callback=None, sym_gen=None):
    """Internal training function on multiple devices.
    This function will also work for single device as well.

    Parameters
    ----------
    symbol : Symbol
        The network configuration.
    ctx : list of Context
        The training devices.
    arg_names: list of str
        Name of all arguments of the network.
    param_names: list of str
        Name of all trainable parameters of the network.
    aux_names: list of str
        Name of all auxiliary states of the network.
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    begin_epoch : int
        The begining training epoch.
    end_epoch : int
        The end training epoch.
    epoch_size : int, optional
        Number of batches in a epoch. In default, it is set to
        ``ceil(num_train_examples / batch_size)``.
    optimizer : Optimizer
        The optimization algorithm
    train_data : DataIter
        Training data iterator.
    eval_data : DataIter
        Validation data iterator.
    eval_metric : EvalMetric
        An evaluation function or a list of evaluation functions.
    epoch_end_callback : callable(epoch, symbol, arg_params, aux_states)
        A callback that is invoked at end of each epoch.
        This can be used to checkpoint model each epoch.
    batch_end_callback : callable(BatchEndParams)
        A callback that is invoked at end of each batch.
        This can be used to measure speed, get result from evaluation metric. etc.
    kvstore : KVStore
        The KVStore.
    update_on_kvstore : bool
        Whether or not perform weight updating on kvstore.
    logger : logging logger
        When not specified, default logger will be used.
    work_load_list : list of float or int, optional
        The list of work load for different devices,
        in the same order as ``ctx``.
    monitor : Monitor, optional
        Monitor installed to executor,
        for monitoring outputs, weights, and gradients for debugging.
    Notes
    -----
    - This function will inplace update the NDArrays in `arg_params` and `aux_states`.
    """
    if logger is None:
        logger = logging
    executor_manager = DataParallelExecutorManager(symbol=symbol,
                                                   sym_gen=sym_gen,
                                                   ctx=ctx,
                                                   train_data=train_data,
                                                   param_names=param_names,
                                                   arg_names=arg_names,
                                                   aux_names=aux_names,
                                                   work_load_list=work_load_list,
                                                   logger=logger)
    if monitor:
        executor_manager.install_monitor(monitor)

    executor_manager.set_params(arg_params, aux_params)

    if not update_on_kvstore:
        updater = get_updater(optimizer)

    if kvstore:
        _initialize_kvstore(kvstore=kvstore,
                            param_arrays=executor_manager.param_arrays,
                            arg_params=arg_params,
                            param_names=executor_manager.param_names,
                            update_on_kvstore=update_on_kvstore)

    if update_on_kvstore:
        kvstore.set_optimizer(optimizer)

    # Now start training
    train_data.reset()
    for epoch in range(begin_epoch, end_epoch):
        # Training phase
        tic = time.time()
        eval_metric.reset()
        nbatch = 0
        # Iterate over training data.
        while True:
            do_reset = True
            for data_batch in train_data:
                executor_manager.load_data_batch(data_batch)

                if monitor is not None:
                    monitor.tic()

                executor_manager.forward(is_train=True)
                executor_manager.backward()

                if update_on_kvstore:
                    _update_params_on_kvstore(executor_manager.param_arrays,
                                              executor_manager.grad_arrays,
                                              kvstore, executor_manager.param_names)
                else:
                    _update_params(executor_manager.param_arrays,
                                   executor_manager.grad_arrays,
                                   updater=updater,
                                   num_device=len(ctx),
                                   kvstore=kvstore,
                                   param_names=executor_manager.param_names)

                if monitor is not None:
                    monitor.toc_print()

                # evaluate at end, so we can lazy copy
                executor_manager.update_metric(eval_metric, data_batch.label)

                nbatch += 1
                # batch callback (for print purpose)
                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch,
                                                     nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    _multiple_callbacks(batch_end_callback, batch_end_params)

                # this epoch is done possibly earlier
                if epoch_size is not None and nbatch >= epoch_size:
                    do_reset = False
                    break

            if do_reset:
                logger.info('Epoch[%d] Resetting Data Iterator', epoch)
                train_data.reset()

            # this epoch is done
            if epoch_size is None or nbatch >= epoch_size:
                break

        toc = time.time()
        logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))

        if epoch_end_callback or epoch + 1 == end_epoch:
            executor_manager.copy_to(arg_params, aux_params)

        _multiple_callbacks(epoch_end_callback, epoch, symbol, arg_params, aux_params)

        # evaluation
        if eval_data:
            eval_metric.reset()
            eval_data.reset()
            total_num_batch = 0
            for i, eval_batch in enumerate(eval_data):
                executor_manager.load_data_batch(eval_batch)
                executor_manager.forward(is_train=False)
                executor_manager.update_metric(eval_metric, eval_batch.label)
                if eval_batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch,
                                                     nbatch=i,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    _multiple_callbacks(eval_batch_end_callback, batch_end_params)
                total_num_batch += 1
            if eval_end_callback is not None:
                eval_end_params = BatchEndParam(epoch=epoch,
                                                nbatch=total_num_batch,
                                                eval_metric=eval_metric,
                                                locals=locals())
                _multiple_callbacks(eval_end_callback, eval_end_params)
            eval_data.reset()
    # end of all epochs
    return


def save_checkpoint(prefix, epoch, symbol, arg_params, aux_params):
    """Checkpoint the model data into file.

    Parameters
    ----------
    prefix : str
        Prefix of model name.
    epoch : int
        The epoch number of the model.
    symbol : Symbol
        The input Symbol.
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    Notes
    -----
    - ``prefix-symbol.json`` will be saved for symbol.
    - ``prefix-epoch.params`` will be saved for parameters.
    """
    if symbol is not None:
        symbol.save('%s-symbol.json' % prefix)

    save_dict = {('arg:%s' % k) : v.as_in_context(cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k) : v.as_in_context(cpu()) for k, v in aux_params.items()})
    param_name = '%s-%04d.params' % (prefix, epoch)
    nd.save(param_name, save_dict)
    logging.info('Saved checkpoint to \"%s\"', param_name)


def load_checkpoint(prefix, epoch):
    """Load model checkpoint from file.

    Parameters
    ----------
    prefix : str
        Prefix of model name.
    epoch : int
        Epoch number of model we would like to load.

    Returns
    -------
    symbol : Symbol
        The symbol configuration of computation network.
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.

    Notes
    -----
    - Symbol will be loaded from ``prefix-symbol.json``.
    - Parameters will be loaded from ``prefix-epoch.params``.
    """
    symbol = sym.load('%s-symbol.json' % prefix)
    save_dict = nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return (symbol, arg_params, aux_params)

from .callback import LogValidationMetricsCallback # pylint: disable=wrong-import-position

class FeedForward(BASE_ESTIMATOR):
    """Model class of MXNet for training and predicting feedforward nets.
    This class is designed for a single-data single output supervised network.

    Parameters
    ----------
    symbol : Symbol
        The symbol configuration of computation network.
    ctx : Context or list of Context, optional
        The device context of training and prediction.
        To use multi GPU training, pass in a list of gpu contexts.
    num_epoch : int, optional
        Training parameter, number of training epochs(epochs).
    epoch_size : int, optional
        Number of batches in a epoch. In default, it is set to
        ``ceil(num_train_examples / batch_size)``.
    optimizer : str or Optimizer, optional
        Training parameter, name or optimizer object for training.
    initializer : initializer function, optional
        Training parameter, the initialization scheme used.
    numpy_batch_size : int, optional
        The batch size of training data.
        Only needed when input array is numpy.
    arg_params : dict of str to NDArray, optional
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray, optional
        Model parameter, dict of name to NDArray of net's auxiliary states.
    allow_extra_params : boolean, optional
        Whether allow extra parameters that are not needed by symbol
        to be passed by aux_params and ``arg_params``.
        If this is True, no error will be thrown when ``aux_params`` and ``arg_params``
        contain more parameters than needed.
    begin_epoch : int, optional
        The begining training epoch.
    kwargs : dict
        The additional keyword arguments passed to optimizer.
    """
    def __init__(self, symbol, ctx=None,
                 num_epoch=None, epoch_size=None, optimizer='sgd',
                 initializer=Uniform(0.01),
                 numpy_batch_size=128,
                 arg_params=None, aux_params=None,
                 allow_extra_params=False,
                 begin_epoch=0,
                 **kwargs):
        warnings.warn(
            '\033[91mmxnet.model.FeedForward has been deprecated. ' + \
            'Please use mxnet.mod.Module instead.\033[0m',
            DeprecationWarning, stacklevel=2)

        if isinstance(symbol, sym.Symbol):
            self.symbol = symbol
            self.sym_gen = None
        else:
            assert(callable(symbol))
            self.symbol = None
            self.sym_gen = symbol

        # model parameters
        self.arg_params = arg_params
        self.aux_params = aux_params
        self.allow_extra_params = allow_extra_params

        self.argument_checked = False
        if self.sym_gen is None:
            self._check_arguments()

        # basic configuration
        if ctx is None:
            ctx = [cpu()]
        elif isinstance(ctx, Context):
            ctx = [ctx]
        self.ctx = ctx
        # training parameters
        self.num_epoch = num_epoch
        self.epoch_size = epoch_size
        self.kwargs = kwargs.copy()
        self.optimizer = optimizer
        self.initializer = initializer
        self.numpy_batch_size = numpy_batch_size
        # internal helper state
        self._pred_exec = None
        self.begin_epoch = begin_epoch

    def _check_arguments(self):
        """verify the argument of the default symbol and user provided parameters"""
        if self.argument_checked:
            return

        assert(self.symbol is not None)
        self.argument_checked = True

        # check if symbol contain duplicated names.
        _check_arguments(self.symbol)
        # rematch parameters to delete useless ones
        if self.allow_extra_params:
            if self.arg_params:
                arg_names = set(self.symbol.list_arguments())
                self.arg_params = {k : v for k, v in self.arg_params.items()
                                   if k in arg_names}
            if self.aux_params:
                aux_names = set(self.symbol.list_auxiliary_states())
                self.aux_params = {k : v for k, v in self.aux_params.items()
                                   if k in aux_names}


    @staticmethod
    def _is_data_arg(name):
        """Check if name is a data argument."""
        return name.endswith('data') or name.endswith('label')

    def _init_params(self, inputs, overwrite=False):
        """Initialize weight parameters and auxiliary states."""
        inputs = [x if isinstance(x, DataDesc) else DataDesc(*x) for x in inputs]
        input_shapes = {item.name: item.shape for item in inputs}
        arg_shapes, _, aux_shapes = self.symbol.infer_shape(**input_shapes)
        assert arg_shapes is not None
        input_dtypes = {item.name: item.dtype for item in inputs}
        arg_dtypes, _, aux_dtypes = self.symbol.infer_type(**input_dtypes)
        assert arg_dtypes is not None

        arg_names = self.symbol.list_arguments()
        input_names = input_shapes.keys()
        param_names = [key for key in arg_names if key not in input_names]
        aux_names = self.symbol.list_auxiliary_states()

        param_name_attrs = [x for x in zip(arg_names, arg_shapes, arg_dtypes)
                            if x[0] in param_names]
        arg_params = {k : nd.zeros(shape=s, dtype=t)
                      for k, s, t in param_name_attrs}
        aux_name_attrs = [x for x in zip(aux_names, aux_shapes, aux_dtypes)
                          if x[0] in aux_names]
        aux_params = {k : nd.zeros(shape=s, dtype=t)
                      for k, s, t in aux_name_attrs}

        for k, v in arg_params.items():
            if self.arg_params and k in self.arg_params and (not overwrite):
                arg_params[k][:] = self.arg_params[k][:]
            else:
                self.initializer(k, v)

        for k, v in aux_params.items():
            if self.aux_params and k in self.aux_params and (not overwrite):
                aux_params[k][:] = self.aux_params[k][:]
            else:
                self.initializer(k, v)

        self.arg_params = arg_params
        self.aux_params = aux_params
        return (arg_names, list(param_names), aux_names)

    def __getstate__(self):
        this = self.__dict__.copy()
        this['_pred_exec'] = None
        return this

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _init_predictor(self, input_shapes, type_dict=None):
        """Initialize the predictor module for running prediction."""
        if self._pred_exec is not None:
            arg_shapes, _, _ = self.symbol.infer_shape(**dict(input_shapes))
            assert arg_shapes is not None, "Incomplete input shapes"
            pred_shapes = [x.shape for x in self._pred_exec.arg_arrays]
            if arg_shapes == pred_shapes:
                return
        # for now only use the first device
        pred_exec = self.symbol.simple_bind(
            self.ctx[0], grad_req='null', type_dict=type_dict, **dict(input_shapes))
        pred_exec.copy_params_from(self.arg_params, self.aux_params)

        _check_arguments(self.symbol)
        self._pred_exec = pred_exec

    def _init_iter(self, X, y, is_train):
        """Initialize the iterator given input."""
        if isinstance(X, (np.ndarray, nd.NDArray)):
            if y is None:
                if is_train:
                    raise ValueError('y must be specified when X is numpy.ndarray')
                else:
                    y = np.zeros(X.shape[0])
            if not isinstance(y, (np.ndarray, nd.NDArray)):
                raise TypeError('y must be ndarray when X is numpy.ndarray')
            if X.shape[0] != y.shape[0]:
                raise ValueError("The numbers of data points and labels not equal")
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.flatten()
            if y.ndim != 1:
                raise ValueError("Label must be 1D or 2D (with 2nd dimension being 1)")
            if is_train:
                return io.NDArrayIter(X, y, min(X.shape[0], self.numpy_batch_size),
                                      shuffle=is_train, last_batch_handle='roll_over')
            else:
                return io.NDArrayIter(X, y, min(X.shape[0], self.numpy_batch_size), shuffle=False)
        if not isinstance(X, io.DataIter):
            raise TypeError('X must be DataIter, NDArray or numpy.ndarray')
        return X

    def _init_eval_iter(self, eval_data):
        """Initialize the iterator given eval_data."""
        if eval_data is None:
            return eval_data
        if isinstance(eval_data, (tuple, list)) and len(eval_data) == 2:
            if eval_data[0] is not None:
                if eval_data[1] is None and isinstance(eval_data[0], io.DataIter):
                    return eval_data[0]
                input_data = (np.array(eval_data[0]) if isinstance(eval_data[0], list)
                              else eval_data[0])
                input_label = (np.array(eval_data[1]) if isinstance(eval_data[1], list)
                               else eval_data[1])
                return self._init_iter(input_data, input_label, is_train=True)
            else:
                raise ValueError("Eval data is NONE")
        if not isinstance(eval_data, io.DataIter):
            raise TypeError('Eval data must be DataIter, or ' \
                            'NDArray/numpy.ndarray/list pair (i.e. tuple/list of length 2)')
        return eval_data

    def predict(self, X, num_batch=None, return_data=False, reset=True):
        """Run the prediction, always only use one device.

        Parameters
        ----------
        X : mxnet.DataIter
        num_batch : int or None
            The number of batch to run. Go though all batches if ``None``.
        Returns
        -------
        y : numpy.ndarray or a list of numpy.ndarray if the network has multiple outputs.
            The predicted value of the output.
        """
        X = self._init_iter(X, None, is_train=False)

        if reset:
            X.reset()
        data_shapes = X.provide_data
        data_names = [x[0] for x in data_shapes]
        type_dict = dict((key, value.dtype) for (key, value) in self.arg_params.items())
        for x in X.provide_data:
            if isinstance(x, DataDesc):
                type_dict[x.name] = x.dtype
            else:
                type_dict[x[0]] = mx_real_t

        self._init_predictor(data_shapes, type_dict)
        batch_size = X.batch_size
        data_arrays = [self._pred_exec.arg_dict[name] for name in data_names]
        output_list = [[] for _ in range(len(self._pred_exec.outputs))]
        if return_data:
            data_list = [[] for _ in X.provide_data]
            label_list = [[] for _ in X.provide_label]

        i = 0
        for batch in X:

            _load_data(batch, data_arrays)
            self._pred_exec.forward(is_train=False)
            padded = batch.pad
            real_size = batch_size - padded

            for o_list, o_nd in zip(output_list, self._pred_exec.outputs):
                o_list.append(o_nd[0:real_size].asnumpy())

            if return_data:
                for j, x in enumerate(batch.data):
                    data_list[j].append(x[0:real_size].asnumpy())
                for j, x in enumerate(batch.label):
                    label_list[j].append(x[0:real_size].asnumpy())
            i += 1
            if num_batch is not None and i == num_batch:
                break

        outputs = [np.concatenate(x) for x in output_list]
        if len(outputs) == 1:
            outputs = outputs[0]

        if return_data:
            data = [np.concatenate(x) for x in data_list]
            label = [np.concatenate(x) for x in label_list]
            if len(data) == 1:
                data = data[0]
            if len(label) == 1:
                label = label[0]
            return outputs, data, label
        else:
            return outputs

    def score(self, X, eval_metric='acc', num_batch=None, batch_end_callback=None, reset=True):
        """Run the model given an input and calculate the score
        as assessed by an evaluation metric.

        Parameters
        ----------
        X : mxnet.DataIter
        eval_metric : metric.metric
            The metric for calculating score.
        num_batch : int or None
            The number of batches to run. Go though all batches if ``None``.
        Returns
        -------
        s : float
            The final score.
        """
        # setup metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        X = self._init_iter(X, None, is_train=False)
        if reset:
            X.reset()

        data_shapes = X.provide_data
        data_names = [x[0] for x in data_shapes]
        type_dict = dict((key, value.dtype) for (key, value) in self.arg_params.items())
        for x in X.provide_data:
            if isinstance(x, DataDesc):
                type_dict[x.name] = x.dtype
            else:
                type_dict[x[0]] = mx_real_t

        self._init_predictor(data_shapes, type_dict)
        data_arrays = [self._pred_exec.arg_dict[name] for name in data_names]

        for i, batch in enumerate(X):
            if num_batch is not None and i == num_batch:
                break
            _load_data(batch, data_arrays)
            self._pred_exec.forward(is_train=False)
            eval_metric.update(batch.label, self._pred_exec.outputs)

            if batch_end_callback is not None:
                batch_end_params = BatchEndParam(epoch=0,
                                                 nbatch=i,
                                                 eval_metric=eval_metric,
                                                 locals=locals())
                _multiple_callbacks(batch_end_callback, batch_end_params)
        return eval_metric.get()[1]

    def fit(self, X, y=None, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local', logger=None,
            work_load_list=None, monitor=None, eval_end_callback=LogValidationMetricsCallback(),
            eval_batch_end_callback=None):
        """Fit the model.

        Parameters
        ----------
        X : DataIter, or numpy.ndarray/NDArray
            Training data. If `X` is a `DataIter`, the name or (if name not available)
            the position of its outputs should match the corresponding variable
            names defined in the symbolic graph.
        y : numpy.ndarray/NDArray, optional
            Training set label.
            If X is ``numpy.ndarray`` or `NDArray`, `y` is required to be set.
            While y can be 1D or 2D (with 2nd dimension as 1), its first dimension must be
            the same as `X`, i.e. the number of data points and labels should be equal.
        eval_data : DataIter or numpy.ndarray/list/NDArray pair
            If eval_data is numpy.ndarray/list/NDArray pair,
            it should be ``(valid_data, valid_label)``.
        eval_metric : metric.EvalMetric or str or callable
            The evaluation metric. This could be the name of evaluation metric
            or a custom evaluation function that returns statistics
            based on a minibatch.
        epoch_end_callback : callable(epoch, symbol, arg_params, aux_states)
            A callback that is invoked at end of each epoch.
            This can be used to checkpoint model each epoch.
        batch_end_callback: callable(epoch)
            A callback that is invoked at end of each batch for purposes of printing.
        kvstore: KVStore or str, optional
           The KVStore or a string kvstore type: 'local', 'dist_sync', 'dist_async'
           In default uses 'local', often no need to change for single machiine.
        logger : logging logger, optional
            When not specified, default logger will be used.
        work_load_list : float or int, optional
            The list of work load for different devices,
            in the same order as `ctx`.

        Note
        ----
        KVStore behavior
        - 'local', multi-devices on a single machine, will automatically choose best type.
        - 'dist_sync', multiple machines communicating via BSP.
        - 'dist_async', multiple machines with asynchronous communication.
        """

        data = self._init_iter(X, y, is_train=True)
        eval_data = self._init_eval_iter(eval_data)

        if self.sym_gen:
            self.symbol = self.sym_gen(data.default_bucket_key) # pylint: disable=no-member
            self._check_arguments()
        self.kwargs["sym"] = self.symbol

        arg_names, param_names, aux_names = \
                self._init_params(data.provide_data+data.provide_label)

        # setup metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        # create kvstore
        (kvstore, update_on_kvstore) = _create_kvstore(
            kvstore, len(self.ctx), self.arg_params)

        param_idx2name = {}
        if update_on_kvstore:
            param_idx2name.update(enumerate(param_names))
        else:
            for i, n in enumerate(param_names):
                for k in range(len(self.ctx)):
                    param_idx2name[i*len(self.ctx)+k] = n
        self.kwargs["param_idx2name"] = param_idx2name

        # init optmizer
        if isinstance(self.optimizer, str):
            batch_size = data.batch_size
            if kvstore and 'dist' in kvstore.type and not '_async' in kvstore.type:
                batch_size *= kvstore.num_workers
            optimizer = opt.create(self.optimizer,
                                   rescale_grad=(1.0/batch_size),
                                   **(self.kwargs))
        elif isinstance(self.optimizer, opt.Optimizer):
            optimizer = self.optimizer

        # do training
        _train_multi_device(self.symbol, self.ctx, arg_names, param_names, aux_names,
                            self.arg_params, self.aux_params,
                            begin_epoch=self.begin_epoch, end_epoch=self.num_epoch,
                            epoch_size=self.epoch_size,
                            optimizer=optimizer,
                            train_data=data, eval_data=eval_data,
                            eval_metric=eval_metric,
                            epoch_end_callback=epoch_end_callback,
                            batch_end_callback=batch_end_callback,
                            kvstore=kvstore, update_on_kvstore=update_on_kvstore,
                            logger=logger, work_load_list=work_load_list, monitor=monitor,
                            eval_end_callback=eval_end_callback,
                            eval_batch_end_callback=eval_batch_end_callback,
                            sym_gen=self.sym_gen)


    def save(self, prefix, epoch=None):
        """Checkpoint the model checkpoint into file.
        You can also use `pickle` to do the job if you only work on Python.
        The advantage of `load` and `save` (as compared to `pickle`) is that
        the resulting file can be loaded from other MXNet language bindings.
        One can also directly `load`/`save` from/to cloud storage(S3, HDFS)

        Parameters
        ----------
        prefix : str
            Prefix of model name.

        Notes
        -----
        - ``prefix-symbol.json`` will be saved for symbol.
        - ``prefix-epoch.params`` will be saved for parameters.
        """
        if epoch is None:
            epoch = self.num_epoch
        assert epoch is not None
        save_checkpoint(prefix, epoch, self.symbol, self.arg_params, self.aux_params)

    @staticmethod
    def load(prefix, epoch, ctx=None, **kwargs):
        """Load model checkpoint from file.

        Parameters
        ----------
        prefix : str
            Prefix of model name.
        epoch : int
            epoch number of model we would like to load.
        ctx : Context or list of Context, optional
            The device context of training and prediction.
        kwargs : dict
            Other parameters for model, including `num_epoch`, optimizer and `numpy_batch_size`.

        Returns
        -------
        model : FeedForward
            The loaded model that can be used for prediction.

        Notes
        -----
        - ``prefix-symbol.json`` will be saved for symbol.
        - ``prefix-epoch.params`` will be saved for parameters.
        """
        symbol, arg_params, aux_params = load_checkpoint(prefix, epoch)
        return FeedForward(symbol, ctx=ctx,
                           arg_params=arg_params, aux_params=aux_params,
                           begin_epoch=epoch,
                           **kwargs)

    @staticmethod
    def create(symbol, X, y=None, ctx=None,
               num_epoch=None, epoch_size=None, optimizer='sgd', initializer=Uniform(0.01),
               eval_data=None, eval_metric='acc',
               epoch_end_callback=None, batch_end_callback=None,
               kvstore='local', logger=None, work_load_list=None,
               eval_end_callback=LogValidationMetricsCallback(),
               eval_batch_end_callback=None, **kwargs):
        """Functional style to create a model.
        This function is more consistent with functional
        languages such as R, where mutation is not allowed.

        Parameters
        ----------
        symbol : Symbol
            The symbol configuration of a computation network.
        X : DataIter
            Training data.
        y : numpy.ndarray, optional
            If `X` is a ``numpy.ndarray``, `y` must be set.
        ctx : Context or list of Context, optional
            The device context of training and prediction.
            To use multi-GPU training, pass in a list of GPU contexts.
        num_epoch : int, optional
            The number of training epochs(epochs).
        epoch_size : int, optional
            Number of batches in a epoch. In default, it is set to
            ``ceil(num_train_examples / batch_size)``.
        optimizer : str or Optimizer, optional
            The name of the chosen optimizer, or an optimizer object, used for training.
        initializier : initializer function, optional
            The initialization scheme used.
        eval_data : DataIter or numpy.ndarray pair
            If `eval_set` is ``numpy.ndarray`` pair, it should
            be (`valid_data`, `valid_label`).
        eval_metric : metric.EvalMetric or str or callable
            The evaluation metric. Can be the name of an evaluation metric
            or a custom evaluation function that returns statistics
            based on a minibatch.
        epoch_end_callback : callable(epoch, symbol, arg_params, aux_states)
            A callback that is invoked at end of each epoch.
            This can be used to checkpoint model each epoch.
        batch_end_callback: callable(epoch)
            A callback that is invoked at end of each batch for print purposes.
        kvstore: KVStore or str, optional
           The KVStore or a string kvstore type: 'local', 'dist_sync', 'dis_async'.
           Defaults to 'local', often no need to change for single machiine.
        logger : logging logger, optional
            When not specified, default logger will be used.
        work_load_list : list of float or int, optional
            The list of work load for different devices,
            in the same order as `ctx`.
        """
        model = FeedForward(symbol, ctx=ctx, num_epoch=num_epoch,
                            epoch_size=epoch_size,
                            optimizer=optimizer, initializer=initializer, **kwargs)
        model.fit(X, y, eval_data=eval_data, eval_metric=eval_metric,
                  epoch_end_callback=epoch_end_callback,
                  batch_end_callback=batch_end_callback,
                  kvstore=kvstore,
                  logger=logger,
                  work_load_list=work_load_list,
                  eval_end_callback=eval_end_callback,
                  eval_batch_end_callback=eval_batch_end_callback)
        return model
