# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-locals, too-many-arguments
"""Executor manager"""
from __future__ import absolute_import

from . import ndarray as nd
from .context import cpu

import logging

def _split_input_slice(batch_size, work_load_list):
    """Get input slice from the input shape.
    Parameters
    ----------
    batch_size : int
        The number of samples in a mini-batch.
    work_load_list : list of float or int, optional
        The list of work load for different devices,
        in the same order as ctx
    Returns
    -------
    slices : list of slice
        The split slices to get a specific slice.
    Raises
    ------
    ValueError
        If there are two many splits such that some slice can be empty.
    """
    total_work_load = sum(work_load_list)
    batch_num_list = [round(work_load * batch_size / total_work_load)
                      for work_load in work_load_list]
    batch_num_sum = sum(batch_num_list)
    if batch_num_sum < batch_size:
        batch_num_list[-1] += batch_size - batch_num_sum
    slices = []
    end = 0
    for batch_num in batch_num_list:
        begin = int(min((end, batch_size)))
        end = int(min((begin + batch_num, batch_size)))
        if begin >= end:
            raise ValueError('Too many slices such that some splits are empty')
        slices.append(slice(begin, end))
    return slices

def _check_arguments(symbol):
    """Check the argument names of symbol.
    This function checks the duplication of arguments in Symbol.
    The check is done for feedforward net for now.
    Parameters
    ----------
    symbol : Symbol
        The network configuration
    """
    arg_set = set()
    arg_names = symbol.list_arguments()
    for name in arg_names:
        if name in arg_set:
            raise ValueError(('Find duplicated argument name \"%s\", ' +
                              'please make the weight name non-duplicated(using name arguments), ' +
                              'arguments are %s') % (name, str(arg_names)))
        arg_set.add(name)

    aux_set = set()
    aux_names = symbol.list_auxiliary_states()
    for name in aux_names:
        if name in aux_set:
            raise ValueError(
                ('Find duplicated auxiliary param name \"%s\", ' +
                 'please make the weight name non-duplicated(using name arguments), ' +
                 'arguments are %s, auxiliary params are %s'
                ) % (name, str(arg_names), str(aux_names)))
        aux_set.add(name)

def _load_general(data, targets):
    """Load a list of arrays into a list of arrays specified by slices"""
    for d_src, d_targets in zip(data, targets):
        if isinstance(d_targets, nd.NDArray):
            d_src.copyto(d_targets)
        else:
            for slice_idx, d_dst in d_targets:
                d_src[slice_idx].copyto(d_dst)

def _load_data(batch, targets):
    """Load data into sliced arrays"""
    _load_general(batch.data, targets)

def _load_label(batch, targets):
    """Load label into sliced arrays"""
    _load_general(batch.label, targets)

class DataParallelExecutorManager(object):
    """ Helper class to manage multiple executors for data parallelism.
    Parameters
    ----------
    symbol : Symbol
        output symbol
    ctx : list of Context
        devices to run on
    param_names: list of str
        Name of all trainable parameters of the network.
    arg_names: list of str
        Name of all arguments of the network.
    aux_names: list of str
        Name of all auxiliary states of the network.
    train_data : DataIter
        Training data iterator.
    work_load_list : list of float or int, optional
        The list of work load for different devices,
        in the same order as ctx
    logger : logging logger
        When not specified, default logger will be used.
    """
    def __init__(self, symbol, ctx, train_data,
                 param_names, arg_names, aux_names,
                 work_load_list=None, logger=None):
        if logger is None:
            logger = logging
        # preparation
        num_device = len(ctx)
        logger.info('Start training with %s', str(ctx))

        # make sure the architecture is valid
        _check_arguments(symbol)

        if work_load_list is None:
            work_load_list = [1] * num_device
        assert isinstance(work_load_list, list) and len(work_load_list) == num_device, \
            "Invalid settings for work load. "

        slices = _split_input_slice(train_data.batch_size, work_load_list)
        self.slices = slices

        self.data_names = [x[0] for x in train_data.provide_data]
        self.label_names = [x[0] for x in train_data.provide_label]
        self.aux_names = aux_names
        self.param_idx = [i for i in range(len(arg_names)) if arg_names[i] in param_names]
        self.param_names = [arg_names[i] for i in self.param_idx]

        update_dict = {k: 'write' if k in self.param_names else 'null' for k in arg_names}

        self.train_execs = []
        for i in range(len(ctx)):
            data_shapes = {k: tuple([slices[i].stop-slices[i].start] + list(v[1:]))
                           for k, v in train_data.provide_data + train_data.provide_label}
            train_exec = symbol.simple_bind(ctx[i], update_dict, **data_shapes)
            self.train_execs.append(train_exec)

        # data structure

        self.data_arrays = [[(slices[i], e.arg_dict[name]) for i, e in enumerate(self.train_execs)]
                            for name in self.data_names]
        self.label_arrays = [[(slices[i], e.arg_dict[name]) for i, e in enumerate(self.train_execs)]
                             for name in self.label_names]

        self.param_arrays = [[e.arg_arrays[i] for e in self.train_execs]
                             for i in self.param_idx]
        self.grad_arrays = [[e.grad_arrays[i] for e in self.train_execs]
                            for i in self.param_idx]

        self.aux_arrays = [[e.aux_arrays[i] for e in self.train_execs]
                           for i in range(len(aux_names))]

    def install_monitor(self, monitor):
        """ Install monitor on all executors """
        for train_exec in self.train_execs:
            monitor.install(train_exec)

    def set_params(self, arg_params, aux_params):
        """ set parameter and aux values
        Parameters
        ----------
        arg_params : list of NDArray
            source parameter arrays
        aux_params : list of NDArray
            source aux arrays
        """

        for texec in self.train_execs:
            texec.copy_params_from(arg_params, aux_params)

    def copy_to(self, arg_params, aux_params):
        """ Copy data from each executor to `arg_params` and `aux_params`
        Parameters
        ----------
        arg_params : list of NDArray
            target parameter arrays
        aux_params : list of NDArray
            target aux arrays
        Notes
        -----
        - This function will inplace update the NDArrays in arg_params and aux_params.
        """
        for name, block in zip(self.param_names, self.param_arrays):
            weight = sum(w.copyto(cpu()) for w in block) / len(block)
            weight.copyto(arg_params[name])
        for name, block in zip(self.aux_names, self.aux_arrays):
            weight = sum(w.copyto(cpu()) for w in block) / len(block)
            weight.copyto(aux_params[name])

    def load_data_batch(self, data_batch):
        """ load data and labels into arrays """
        _load_data(data_batch, self.data_arrays)
        _load_label(data_batch, self.label_arrays)

    def forward(self, is_train=False):
        """ Perform a forward pass on each executor """
        for texec in self.train_execs:
            texec.forward(is_train=is_train)

    def backward(self):
        """ Perform a backward pass on each executor """
        for texec in self.train_execs:
            texec.backward()

    def update_metric(self, metric, labels):
        """ Update evaluation metric with label and current outputs """
        for texec, islice in zip(self.train_execs, self.slices):
            labels_slice = [label[islice] for label in labels]
            metric.update(labels_slice, texec.outputs)

