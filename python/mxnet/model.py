# pylint: skip-file
import numpy as np
import time

from .io import DataIter
from .context import Context
from .ndarray import empty, zeros
from .initializer import Xavier
from .symbol import Symbol
from .optimizer import get_optimizer

Base = object
try:
    from sklearn.base import BaseEstimator
    Base = BaseEstimator
except ImportError:
    SKLEARN_INSTALLED = False


class MXNetModel(object):
    """MXNet model"""
    def __init__(self, ctx, symbol, num_round, batch_size, optimizer="sgd", initializer=Xavier(), **kwargs):
        """Constructor

        Parameter
        ---------
        ctx: Context or list of Context
            running context for model, if is a list, run a multiply device
        symbol: Symbol
            symbol of the model
        num_round: int
            training num round
        batch_size: int
            batch size
        optimizer: str
            optimizer used to train the model
        initializer: Initializer
            initializer used to initialize weight
        kwargs: dict
            optimizer arguments and input data shape
        """
        if not isinstance(symbol, Symbol):
            raise TypeError("symbol")
        if num_round <= 0:
            raise ValueError("num_round must be greater than 0")
        self.ctx = ctx
        self.optimizer = get_optimizer(name=optimizer, batch_size=batch_size, **kwargs)
        self.num_round = num_round
        self.initializer = initializer
        self.shape_dict = kwargs
        self.symbol = symbol
        # check shape and batch size
        arg_shapes, out_shapes, aux_shapes = self.symbol.infer_shape(**kwargs)
        if arg_shapes == None:
            raise ValueError("input shape is incomplete")

    def fit(self, X, y=None, eval_set=None, eval_metric=None):
        """fit the model

        Parameter
        ---------
        X: DataIter or numpy.ndarray(TODO)
            training data
        y: None or numpy.ndarray
            if X is DataIter no need to set (use None)
            if X is numpy.ndarray y is required to set
        eval_set: DataIter or numpy.ndarray pair (TODO)
            if eval_set is numpy.ndarray pair, it should be (valid_data, valid_label)
        eval_metric: function
        """
        self.executor = self.symbol.simple_bind(ctx=self.ctx, **self.shape_dict)
        # init
        arg_narrays, grad_narrays = self.executor.list_arguments()
        inputs = dict(zip(self.symbol.list_arguments(), arg_narrays))
        arg_blocks = list(zip(arg_narrays, grad_narrays, self.symbol.list_arguments()))
        # only support 1 output now
        # find label
        label_node_name = ""
        data_node_name = ""
        for name, ndarray in inputs.items():
            if "label" in name:
                label_node_name = name
            if "data" in name:
                data_node_name = name
        # single output
        out_ndarray = self.executor.outputs[0]
        pred = zeros(out_ndarray.shape)
        for state, narray in inputs.items():
            self.initializer(state, narray)
        for i in range(self.num_round):
            print("Epoch %d:" % i)
            #train
            train_acc = 0.0
            val_acc = 0.0
            train_nbatch = 0
            val_nbatch = 0
            tic = time.time()
            for data, label in X:
                label = label.asnumpy().flatten()
                inputs[label_node_name][:] = label
                inputs[data_node_name][:] = data
                self.executor.forward()
                pred[:] = out_ndarray
                train_nbatch += 1
                self.executor.backward()

                for weight, grad, state in arg_blocks:
                    self.optimizer(weight, grad, state)

                train_acc += eval_metric(pred.asnumpy(), label)
                toc = time.time()
            print("Time: %.3f" % (toc - tic))

            # eval
            for data, label in eval_set:
                label = label.asnumpy().flatten()
                inputs[data_node_name][:] = data
                self.executor.forward()
                val_acc += eval_metric(out_ndarray.asnumpy(), label)
                val_nbatch += 1

            print("Train Acc: ", train_acc / train_nbatch)
            print("Valid Acc: ", val_acc / val_nbatch)
            X.reset()
            eval_set.reset()

    def save(self, path):
        """save model

        Parameter
        ---------
        path: str
            saving path
        """
        raise NotImplementedError("TODO")

    def load(self, path):
        """load model

        Parameter
        ---------
        path: str
            saving path
        """
        raise NotImplementedError("TODO")

    def draw(self, path):
        """draw model

        Parameter
        ---------
        path: str
            saving path
        """
        raise NotImplementedError("TODO")

"""
class MXNetClassifier(MXNetModel):
    def __init__(self, ctx, symbol, optimizer, num_round, batch_size, initializer=xavier, **kwargs):
        super(MXNetClassifier, self).__init__(ctx, symbol, optimizer,
                num_round, batch_size, initializer, **kwargs)

    def predict(self):
        pass
    def predict_proba(self, X):
        pass
    def score(self, X, y):
        pass
"""
