# pylint: disable=fixme, invalid-name, too-many-arguments, too-many-locals
# pylint: disable=too-many-branches, too-many-statements, unused-argument
"""MXNet model module"""
import numpy as np
import time
import logging
from . import io
from . import nd
from . import symbol as sym
from . import optimizer as opt
from . import metric
from .context import Context, cpu
from .initializer import Xavier


BASE_ESTIMATOR = object

try:
    from sklearn.base import BaseEstimator
    BASE_ESTIMATOR = BaseEstimator
except ImportError:
    SKLEARN_INSTALLED = False


def _check_arguments(symbol):
    """Check the argument names of symbol.

    This function checks the duplication of arguments in Symbol.
    The check is done for feedforward net for now.

    Parameters
    ----------
    symbol : Symbol
        The network configuration

    Returns
    -------
    data_index : int
        Index position of data.
    label_index : int
        Index position of label
    """
    arg_names = symbol.list_arguments()
    data_index, label_index = None, None
    arg_set = set()
    for index, name in enumerate(arg_names):
        if name.endswith('label'):
            if label_index is not None:
                raise ValueError('Two arguments with suffix \"label\", ' +
                                 'only accept one label in config for now, '+
                                 'arguments are %s' % str(arg_names))
            label_index = index
        if name.endswith('data'):
            if data_index is not None:
                raise ValueError('Two arguments with suffix \"label\", ' +
                                 'only accept one input data in config for now, ' +
                                 'arguments are %s' % str(arg_names))
            data_index = index
        if name in arg_set:
            raise ValueError(('Find duplicated argument name \"%s\", ' +
                              'please make the weight name non-duplicated(using name arguments), ' +
                              'arguments are %s') % (name, str(arg_names)))
        arg_set.add(name)

    aux_names = symbol.list_auxiliary_states()
    for name in aux_names:
        if name in arg_set:
            raise ValueError(
                ('Find duplicated auxiliary param name \"%s\", ' +
                 'please make the weight name non-duplicated(using name arguments), ' +
                 'arguments are %s, auxiliary params are %s'
                ) % (name, str(arg_names), str(aux_names)))

    return (data_index, label_index)


def _train(symbol, ctx, input_shape,
           arg_params, aux_params,
           begin_round, end_round, optimizer,
           train_data, eval_data=None, eval_metric=None,
           iter_end_callback=None, logger=None):
    """Inernal training function.

    Parameters
    ----------
    symbol : Symbol
        The network configuration

    ctx : list of Context
        The training devices.

    input_shape : tuple
        Shape of input data batch.

    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.

    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.

    begin_round : int
        The begining training iteration.

    end_round : int
        The end training iteration.

    optimizer : Optimizer
        The optimization algorithm

    train_data : DataIter
        Training data iterator.

    eval_data : DataIter
        Validation data iterator.

    eval_metric : EvalMetric
        A evaluation function.

    iter_end_callback : callable(iteration, symbol, arg_params, aux_states)
        A callback that is invoked at end of each iteration.
        This can be used to checkpoint model each iteration.

    logger : logging logger
        When not specified, default logger will be used.

    Notes
    -----
    This function will inplace update the NDArrays in arg_parans and aux_states.
    """
    assert(len(ctx) == 1)
    if logger is None:
        logger = logging
    # bind the symbol
    train_exec = symbol.simple_bind(ctx[0], data=input_shape, grad_req='write')
    arg_names = symbol.list_arguments()
    aux_names = symbol.list_auxiliary_states()
    arg_arrays = train_exec.arg_arrays
    grad_arrays = train_exec.grad_arrays
    aux_arrays = train_exec.aux_arrays
    # copy initialized parameters to executor parameters
    for key, weight in zip(arg_names, arg_arrays):
        if key in arg_params:
            arg_params[key].copyto(weight)
    for key, weight in zip(aux_names, aux_arrays):
        if key in aux_params:
            aux_params[key].copyto(weight)
    # setup helper data structures
    data_index, label_index = _check_arguments(symbol)
    data_array, label_array = arg_arrays[data_index], arg_arrays[label_index]
    out_array = train_exec.outputs[0]
    out_cpu_array = nd.zeros(out_array.shape)
    arg_blocks = zip(arg_arrays, grad_arrays)

    for i in range(begin_round, end_round):
        # training phase
        tic = time.time()
        train_data.reset()
        optimizer.begin_round(i)
        eval_metric.reset()

        for data, label in train_data:
            label.copyto(label_array)
            data.copyto(data_array)
            train_exec.forward()
            out_array.copyto(out_cpu_array)
            train_exec.backward()
            # update the parameters
            for index, block in enumerate(arg_blocks):
                weight, grad = block
                if grad is not None:
                    optimizer.update(index, weight, grad)
            # evaluate at end, so out_cpu_array can lazy copy
            eval_metric.update(out_cpu_array, label)

        name, value = eval_metric.get()
        logger.info('Iteration[%d] Train-%s=%f', i, name, value)
        toc = time.time()
        logger.info('Iteration[%d] Time cost=%.3f', i, (toc - tic))

        # evaluation phase
        if eval_data is not None:
            eval_metric.reset()
            eval_data.reset()
            for data, label in eval_data:
                data.copyto(data_array)
                # TODO(bing): add is_train=False
                train_exec.forward()
                out_array.copyto(out_cpu_array)
                eval_metric.update(out_array, label)

            name, value = eval_metric.get()
            logger.info('Iteration[%d] Validation-%s=%f', i, name, value)

        if iter_end_callback or i + 1 == end_round:
            # copy data back to cpu
            for key, weight in zip(arg_names, arg_arrays):
                if key in arg_params:
                    weight.copyto(arg_params[key])
            for key, arr in zip(aux_names, aux_arrays):
                arr.copyto(aux_params[key])
        if iter_end_callback:
            iter_end_callback(i, symbol, arg_params, aux_params)
    # end of the function
    return


def save_checkpoint(prefix, iteration, symbol, arg_params, aux_params):
    """Checkpoint the model data into file.

    Parameters
    ----------
    prefix : str
        Prefix of model name.

    iteration : int
        The iteration number of the model.

    symbol : Symbol
        The input symbol

    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.

    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.

    Notes
    -----
    - ``prefix-symbol.json`` will be saved for symbol.
    - ``prefix-iteration.params`` will be saved for parameters.
    """
    symbol.save('%s-symbol.json' % prefix)
    save_dict = {('arg:%s' % k) : v for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k) : v for k, v in aux_params.items()})
    param_name = '%s-%04d.params' % (prefix, iteration)
    nd.save(param_name, save_dict)
    logging.info('Saved checkpoint to \"%s\"', param_name)


def load_checkpoint(prefix, iteration):
    """Load model checkpoint from file.

    Parameters
    ----------
    prefix : str
        Prefix of model name.

    iteration : int
        Iteration number of model we would like to load.

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
    - ``prefix-symbol.json`` will be saved for symbol.
    - ``prefix-iteration.params`` will be saved for parameters.
    """
    symbol = sym.load('%s-symbol.json' % prefix)
    save_dict = nd.load('%s-%04d.params' % (prefix, iteration))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return (symbol, arg_params, aux_params)


def do_checkpoint(prefix):
    """Callback to checkpoint the model to prefix every iteration.

    Parameters
    ----------
    prefix : str
        The file prefix to checkpoint to

    Returns
    -------
    callback : function
        The callback function that can be passed as iter_end_callback to fit.
    """
    def _callback(iter_no, s, arg, aux):
        """The checkpoint function."""
        save_checkpoint(prefix, iter_no + 1, s, arg, aux)
    return _callback


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

    num_round : int, optional
        Training parameter, number of training rounds(iterations).

    optimizer : str or Optimizer, optional
        Training parameter, name or optimizer object for training.

    initializier : initializer function, optional
        Training parameter, the initialization scheme used.

    arg_params : dict of str to NDArray, optional
        Model parameter, dict of name to NDArray of net's weights.

    aux_params : dict of str to NDArray, optional
        Model parameter, dict of name to NDArray of net's auxiliary states.

    **kwargs : dict
        The additional keyword arguments passed to optimizer.
    """
    def __init__(self, symbol, ctx=None,
                 num_round=None, optimizer='sgd', initializer=Xavier(),
                 arg_params=None, aux_params=None,
                 **kwargs):
        # basic configuration
        self.symbol = symbol
        if ctx is None:
            ctx = [cpu()]
        elif isinstance(ctx, Context):
            ctx = [ctx]
        self.ctx = ctx
        # training parameters
        self.num_round = num_round
        self.kwargs = kwargs.copy()
        self.optimizer = optimizer
        self.initializer = initializer
        # model parameters
        self.arg_params = arg_params
        self.aux_params = aux_params
        # internal helper state
        self._pred_exec = None
        self._pred_exec_input = None

    @staticmethod
    def _is_data_arg(name):
        """Check if name is a data argument."""
        return name.endswith('data') or name.endswith('label')

    @staticmethod
    def _get_input_shape(data):
        """Get input shape from data iterator."""
        data.reset()
        data.next()
        input_shape = data.getdata().shape
        data.reset()
        return input_shape

    def _init_params(self, input_shape):
        """Use initializer to initialize the parameters."""
        arg_shapes, _, aux_shapes = self.symbol.infer_shape(data=input_shape)
        if self.arg_params is None:
            arg_names = self.symbol.list_arguments()
            self.arg_params = {k : nd.zeros(s) for k, s in list(zip(arg_names, arg_shapes))
                               if not self._is_data_arg(k)}
        if self.aux_params is None:
            aux_names = self.symbol.list_auxiliary_states()
            self.aux_params = {k : nd.zeros(s) for k, s in list(zip(aux_names, aux_shapes))}
        for k, v in self.arg_params.items():
            self.initializer(k, v)
        for k, v in self.aux_params.items():
            self.initializer(k, v)

    def __getstate__(self):
        this = self.__dict__.copy()
        this['_pred_exec'] = None
        return this

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _init_predictor(self, input_shape):
        """Initialize the predictor module for running prediction."""
        if self._pred_exec is not None:
            return
        # for now only use the first device
        pred_exec = self.symbol.simple_bind(
            self.ctx[0], grad_req='null', data=input_shape)

        for name, value in list(zip(self.symbol.list_arguments(), pred_exec.arg_arrays)):
            if not self._is_data_arg(name):
                assert name in self.arg_params
                self.arg_params[name].copyto(value)
        data_index, _ = _check_arguments(self.symbol)
        self._pred_exec_input = pred_exec.arg_arrays[data_index]
        self._pred_exec = pred_exec

    def predict(self, X):
        """Run the prediction, always only use one device.

        Parameters
        ----------
        X : mxnet.DataIter or numpy.ndarray

        Returns
        -------
        y : numpy.ndarray
            The predicted value of the output.
        """
        assert isinstance(X, io.DataIter)
        self._init_predictor(self._get_input_shape(X))
        outputs = []

        X.reset()
        for data, _ in X:
            data.copyto(self._pred_exec_input)
            self._pred_exec.forward()
            outputs.append(self._pred_exec.outputs[0].asnumpy())
        return np.concatenate(outputs)

    def fit(self, X, y=None, eval_data=None, eval_metric='acc',
            iter_end_callback=None, logger=None):
        """fit the model

        Parameters
        ----------
        X : DataIter
            Training data

        y : numpy.ndarray, optional
            If X is numpy.ndarray y is required to set

        eval_data : DataIter or numpy.ndarray pair
            If eval_set is numpy.ndarray pair, it should be (valid_data, valid_label)

        eval_metric : function
            Evaluation metric function.

        iter_end_callback : callable(iteration, symbol, arg_params, aux_states)
            A callback that is invoked at end of each iteration.
            This can be used to checkpoint model each iteration.

        logger : logging logger, optional
            When not specified, default logger will be used.
        """
        input_shape = self._get_input_shape(X)
        if self.arg_params is None:
            self._init_params(input_shape)
        # setup metric
        if isinstance(eval_metric, str):
            eval_metric = metric.create(eval_metric)
        # setup optimizer
        optimizer = self.optimizer
        if isinstance(optimizer, str):
            batch_size = input_shape[0]
            optimizer = opt.create(optimizer, rescale_grad=(1.0/batch_size), **(self.kwargs))
        # do training
        _train(self.symbol, self.ctx, input_shape,
               self.arg_params, self.aux_params,
               begin_round=0, end_round=self.num_round,
               optimizer=optimizer,
               train_data=X, eval_data=eval_data,
               eval_metric=eval_metric,
               iter_end_callback=iter_end_callback,
               logger=logger)

    def save(self, prefix, iteration=None):
        """Checkpoint the model checkpoint into file.

        You can also use pickle to do the job if you only work on python.
        The advantage of load/save is the file is language agnostic.
        This means the file saved using save can be loaded by other language binding of mxnet.
        You also get the benefit being able to directly load/save from cloud storage(S3, HDFS)

        Parameters
        ----------
        prefix : str
            Prefix of model name.

        See Also
        --------
        Symbol.load : the method to load the model back.

        Notes
        -----
        - ``prefix-symbol.json`` will be saved for symbol.
        - ``prefix-iteration.params`` will be saved for parameters.
        """
        if iteration is None:
            iteration = self.num_round
        assert iteration is not None
        save_checkpoint(prefix, iteration, self.symbol, self.arg_params, self.aux_params)

    @staticmethod
    def load(prefix, iteration, ctx=None):
        """Load model checkpoint from file.

        Parameters
        ----------
        prefix : str
            Prefix of model name.

        iteration : int
            Iteration number of model we would like to load.

        ctx : Context or list of Context, optional
            The device context of training and prediction.

        Returns
        -------
        model : FeedForward
            The loaded model that can be used for prediction.

        Notes
        -----
        - ``prefix-symbol.json`` will be saved for symbol.
        - ``prefix-iteration.params`` will be saved for parameters.
        """
        symbol, arg_params, aux_params = load_checkpoint(prefix, iteration)
        return FeedForward(symbol, ctx=ctx,
                           arg_params=arg_params, aux_params=aux_params)

