# pylint: skip-file
import numpy as np
import time
from . import io
from . import nd
from . import optimizer as opt
from . import metric
from .symbol import Symbol
from .context import Context
from .initializer import Xavier


BASE_ESTIMATOR = object

try:
    from sklearn.base import BaseEstimator
    BASE_ESTIMATOR = BaseEstimator
except ImportError:
    SKLEARN_INSTALLED = False


def _train(symbol, ctx, input_shape,
           arg_params, aux_states,
           begin_round, end_round, optimizer,
           train_data, eval_data=None, eval_metric=None,
           iter_end_callback=None, verbose=True):
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

    aux_states : dict of str to NDArray
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

    iter_end_callback : callable(iteration, arg_params, aux_states)
        A callback that is invoked at end of each iteration.
        This can be used to checkpoint model each iteration.

    verbose : boolean
        Whether print message during training.

    Notes
    -----
    This function will inplace update the NDArrays in arg_parans and aux_states.
    """
    assert(len(ctx) == 1)
    # bind the symbol
    train_exec = symbol.simple_bind(ctx[0], data=input_shape, grad_req='write')
    arg_names = symbol.list_arguments()
    aux_names = symbol.list_auxiliary_states()
    arg_arrays = train_exec.arg_arrays
    grad_arrays = train_exec.grad_arrays
    print grad_arrays
    aux_arrays = train_exec.aux_arrays
    # copy initialized parameters to executor parameters
    for key, weight in zip(arg_names, arg_arrays):
        if key in arg_params:
            arg_params[key].copyto(weight)
    for key, weight in zip(aux_names, aux_arrays):
        if key in aux_params:
            aux_params[key].copyto(weight)
    # setup helper data structures
    label_array = None
    data_array = None
    for name, arr in zip(symbol.list_arguments(),  arg_arrays):
        if name.endswith('label'):
            assert label_array is None
            label_array = arr
        if name.endswith('data'):
            assert data_array is None
            data_array = arr
    assert data_array is not None
    assert label_array is not None

    out_array = train_exec.outputs[0]
    out_cpu_array = nd.zeros(out_array.shape)
    arg_blocks = list(zip(arg_names, arg_arrays, grad_arrays))

    for i in range(begin_round, end_round):
        if verbose:
            print("Epoch %d:" % i)
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
            for key, weight, grad in arg_blocks:
                if grad is not None:
                    optimizer.update(key, weight, grad)
            # evaluate at end, so out_cpu_array can lazy copy
            eval_metric.update(out_cpu_array, label)

        name, value = eval_metric.get()
        print ('Train %s:\t%f' % (name, value))

        toc = time.time()
        if verbose:
            print("Time: %.3f" % (toc - tic))
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
            print ('Validation %s:\t%f' % (name, value))

        if iter_end_callback or i + 1 == end_round:
            # copy data back to cpu
            for key, weight, gard in arg_blocks:
                if key in arg_params:
                    weight.copyto(arg_params[key])
            for key, arr in zip(aux_names, aux_states):
                arr.copyto(aux_states[key])
        if iter_end_callback:
            iter_end_callback(i, arg_params, aux_states)
    # end of the function
    return


class FeedForward(BASE_ESTIMATOR):
    """Model class of MXNet for training and predicting feedforward nets.

    This class is designed for a single-data single output supervised network.

    Parameters
    ----------
    symbol : Symbol
        The symbol configuration of computation network.

    ctx : Context or list of Context
        The device context of training and prediction.
        To use multi GPU training, pass in a list of gpu contexts.

    input_shape : tuple
        Shape of input data batch.

    num_round : int, optional
        Training parameter, number of training rounds(iterations).

    optimizer : str or Optimizer, optional
        Training parameter, name or optimizer object for training.

    initializier : initializer function, optional
        Training parameter, the initialization scheme used.

    arg_params : dict of str to NDArray, optional
        Model parameter, dict of name to NDArray of net's weights.

    aux_states : dict of str to NDArray, optional
        Model parameter, dict of name to NDArray of net's auxiliary states.

    **kwargs : dict
        The additional keyword arguments passed to optimizer.
    """
    def __init__(self, symbol, ctx, input_shape,
                 num_round=None, optimizer='sgd', initializer=Xavier(),
                 arg_params=None, aux_states=None,
                 **kwargs):
        # basic configuration
        self.symbol = symbol
        if isinstance(ctx, Context):
            ctx = [ctx]
        self.ctx = ctx
        self.input_shape = input_shape
        # training parameters
        self.num_round = num_round
        if isinstance(optimizer, str):
            batch_size = input_shape[0]
            optimizer = opt.create(optimizer, rescale_grad=(1.0/batch_size), **kwargs)
        self.optimizer = optimizer
        print type(self.optimizer)
        self.initializer = initializer
        # model parameters
        self.arg_params = arg_params
        self.aux_states = aux_states
        # internal helper state
        self._pred_exec = None
        self._pred_exec_input = None

    def _init_params(self):
        """Use initializer to initialize the parameters."""
        is_data_arg = lambda x: x.endswith('data') or x.endswith('label')
        arg_shapes, _, aux_shapes = self.symbol.infer_shape(data=self.input_shape)
        if self.arg_params is None:
            arg_names = self.symbol.list_arguments()
            self.arg_params = {k : nd.zeros(s) for k, s in zip(arg_names, arg_shapes)
                               if not is_data_arg(k)}
        if self.aux_states is None:
            aux_names = self.symbol.list_auxiliary_states()
            self.aux_states = {k : nd.zeros(s) for k, s in zip(aux_names, aux_shapes)}
        for k, v in self.arg_params.items():
            self.initializer(k, v)
        for k, v in self.aux_states.items():
            self.initializer(k, v)

    def _init_predictor(self):
        """Initialize the predictor module for running prediction."""
        if self._pred_exec is not None:
            return
        # for now only use the first device
        pred_exec = self.symbol.simple_bind(
            self.ctx[0], grad_req='null', data=self.input_shape)
        for name, value in zip(self.symbol.list_arguments(), pred_exec.arg_arrays):
            if name not in self.arg_datas:
                assert name in self.arg_params
                self.arg_params[name].copyto(value)
            else:
                assert self._pred_exec_input is None
                self._pred_exec_input = value
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
        self._init_predictor()
        outputs = []
        for data, label in X:
            data.copyto(self.pred_exec_input)
            self._pred_exec.forward()
            outputs.extend(self._pred_exec.outputs[0].asnumpy())
        return np.concatenate(outputs)

    def fit(self, X, y=None, eval_data=None, eval_metric='acc', verbose=True):
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

        verbose : boolean
            Whether print information during training.
        """
        if self.arg_params is None:
            self._init_params()
        if isinstance(eval_metric, str):
            eval_metric = metric.create(eval_metric)

        _train(self.symbol, self.ctx, self.input_shape,
               self.arg_params, self.aux_states,
               begin_round=0, end_round=self.num_round,
               optimizer=self.optimizer,
               train_data=X, eval_data=eval_data,
               eval_metric=eval_metric,
               verbose=verbose)
