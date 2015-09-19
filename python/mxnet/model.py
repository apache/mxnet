# pylint: skip-file
import numpy as np
import time

from .io import DataIter
from .context import Context
from .ndarray import empty, zeros
from .initializer import Initializer
from .symbol import Symbol
Base = object
try:
    from sklearn.base import BaseEstimator
    Base = BaseEstimator
except ImportError:
    SKLEARN_INSTALLED = False


class MXNetModel(object):
    def __init__(self, ctx, symbol, optimizer, num_round, batch_size, initializer=Initializer(init_type="xavier"), **kwargs):
        if not isinstance(symbol, Symbol):
            raise TypeError("symbol")
        if num_round <= 0:
            raise ValueError("num_round must be greater than 0")
        self.ctx = ctx
        self.optimizer = optimizer
        self.num_round = num_round
        self.optimizer.batch_size = batch_size
        self.initializer = initializer
        self.shape_dict = kwargs
        self.symbol = symbol
        # check shape and batch size
        arg_shapes, out_shapes, aux_shapes = self.symbol.infer_shape(**kwargs)
        if arg_shapes == None:
            raise ValueError("input shape is incomplete")

    def fit(self, X, y=None, eval_set=None, eval_metric=None):
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
                # todo(xxx): need perf
                label = label.asnumpy().flatten()
                inputs[label_node_name][:] = label
                inputs[data_node_name][:] = data
                self.executor.forward()
                train_acc += eval_metric(out_ndarray.asnumpy(), label)
                train_nbatch += 1
                self.executor.backward()

                for weight, grad, state in arg_blocks:
                    self.optimizer.update(weight, grad, state)
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

    def save(self):
        raise NotImplementedError("TODO")

    def load(self):
        raise NotImplementedError("TODO")

    def draw(self):
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
