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

# pylint: disable=too-many-instance-attributes, too-many-arguments, unnecessary-pass
"""Provide some handy classes for user to implement a simple computation module
in Python easily.
"""
import logging

from .base_module import BaseModule
from ..initializer import Uniform
from .. import ndarray as nd

class PythonModule(BaseModule):
    """A convenient module class that implements many of the module APIs as
    empty functions.

    Parameters
    ----------
    data_names : list of str
        Names of the data expected by the module.
    label_names : list of str
        Names of the labels expected by the module. Could be ``None`` if the
        module does not need labels.
    output_names : list of str
        Names of the outputs.
    """
    def __init__(self, data_names, label_names, output_names, logger=logging):
        super(PythonModule, self).__init__(logger=logger)

        if isinstance(data_names, tuple):
            data_names = list(data_names)
        if isinstance(label_names, tuple):
            label_names = list(label_names)

        self._data_names = data_names
        self._label_names = label_names
        self._output_names = output_names

        self._data_shapes = None
        self._label_shapes = None
        self._output_shapes = None

    ################################################################################
    # Symbol information
    ################################################################################
    @property
    def data_names(self):
        """A list of names for data required by this module."""
        return self._data_names

    @property
    def output_names(self):
        """A list of names for the outputs of this module."""
        return self._output_names

    ################################################################################
    # Input/Output information
    ################################################################################
    @property
    def data_shapes(self):
        """A list of (name, shape) pairs specifying the data inputs to this module."""
        return self._data_shapes

    @property
    def label_shapes(self):
        """A list of (name, shape) pairs specifying the label inputs to this module.
        If this module does not accept labels -- either it is a module without loss
        function, or it is not bound for training, then this should return an empty
        list ``[]```.
        """
        return self._label_shapes

    @property
    def output_shapes(self):
        """A list of (name, shape) pairs specifying the outputs of this module."""
        return self._output_shapes

    ################################################################################
    # Parameters of a module
    ################################################################################
    def get_params(self):
        """Gets parameters, those are potentially copies of the the actual parameters used
        to do computation on the device. Subclass should override this method if contains
        parameters.

        Returns
        -------
        ``({}, {})``, a pair of empty dict.
        """
        return (dict(), dict())

    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False, allow_extra=False):
        """Initializes the parameters and auxiliary states. By default this function
        does nothing. Subclass should override this method if contains parameters.

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
            If ``True``, will force re-initialize even if already initialized.
        allow_extra : boolean, optional
            Whether allow extra parameters that are not needed by symbol.
            If this is True, no error will be thrown when arg_params or aux_params
            contain extra parameters that is not needed by the executor.
        """
        pass

    def update(self):
        """Updates parameters according to the installed optimizer and the gradients computed
        in the previous forward-backward batch. Currently we do nothing here. Subclass should
        override this method if contains parameters.
        """
        pass

    def update_metric(self, eval_metric, labels, pre_sliced=False):
        """Evaluates and accumulates evaluation metric on outputs of the last forward computation.
        Subclass should override this method if needed.

        Parameters
        ----------
        eval_metric : EvalMetric
        labels : list of NDArray
            Typically ``data_batch.label``.
        """
        if self._label_shapes is None:
            # since we do not need labels, we are probably not a module with a loss
            # function or predictions, so just ignore this call
            return

        if pre_sliced:
            raise RuntimeError("PythonModule does not support presliced labels")

        # by default we expect our outputs are some scores that could be evaluated
        eval_metric.update(labels, self.get_outputs())

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
        data_shapes : list of (str, tuple)
            Typically is ``data_iter.provide_data``.
        label_shapes : list of (str, tuple)
            Typically is ``data_iter.provide_label``.
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
        """
        if self.binded and not force_rebind:
            self.logger.warning('Already bound, ignoring bind()')
            return

        assert grad_req == 'write', "Python module only support write gradient"
        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad

        assert len(data_shapes) == len(self._data_names)
        assert [x[0] for x in data_shapes] == self._data_names
        self._data_shapes = data_shapes

        self._label_shapes = label_shapes
        if label_shapes is not None:
            assert self._label_names is not None
            assert len(self._label_names) == len(label_shapes)
            assert [x[0] for x in label_shapes] == self._label_names

        self._output_shapes = self._compute_output_shapes()

    def _compute_output_shapes(self):
        """The subclass should implement this method to compute the shape of
        outputs. This method can assume that the ``data_shapes`` and ``label_shapes``
        are already initialized.
        """
        raise NotImplementedError()

    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),), force_init=False):
        """Installs and initializes optimizers. By default we do nothing. Subclass should
        override this method if needed.

        Parameters
        ----------
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `(('learning_rate', 0.01),)`. The default value is not a dictionary,
            just to avoid pylint warning of dangerous default values.
        force_init : bool
            Default `False`, indicating whether we should force re-initializing the
            optimizer in the case an optimizer is already installed.
        """
        pass


class PythonLossModule(PythonModule):
    """A convenient module class that implements many of the module APIs as
    empty functions.

    Parameters
    ----------
    name : str
        Names of the module. The outputs will be named `[name + '_output']`.
    data_names : list of str
        Defaults to ``['data']``. Names of the data expected by this module.
        Should be a list of only one name.
    label_names : list of str
        Default ``['softmax_label']``. Names of the labels expected by the module.
        Should be a list of only one name.
    grad_func : function
        Optional. If not ``None``, should be a function that takes `scores`
        and `labels`, both of type `NDArray`, and return the gradients with
        respect to the scores according to this loss function. The return
        value could be a numpy array or an `NDArray`.
    """
    def __init__(self, name='pyloss', data_names=('data',), label_names=('softmax_label',),
                 logger=logging, grad_func=None):
        super(PythonLossModule, self).__init__(data_names, label_names,
                                               [name + '_output'], logger=logger)
        self._name = name
        assert len(data_names) == 1
        assert len(label_names) == 1

        self._scores = None
        self._labels = None
        self._scores_grad = None

        if grad_func is not None:
            assert callable(grad_func)
        self._grad_func = grad_func

    def _compute_output_shapes(self):
        """Computes the shapes of outputs. As a loss module with outputs, we simply
        output whatever we receive as inputs (i.e. the scores).
        """
        return [(self._name + '_output', self._data_shapes[0][1])]

    def forward(self, data_batch, is_train=None):
        """Forward computation. Here we do nothing but to keep a reference to
        the scores and the labels so that we can do backward computation.

        Parameters
        ----------
        data_batch : DataBatch
            Could be anything with similar API implemented.
        is_train : bool
            Default is ``None``, which means `is_train` takes the value of ``self.for_training``.
        """
        self._scores = data_batch.data[0]

        if is_train is None:
            is_train = self.for_training

        if is_train:
            self._labels = data_batch.label[0]

    def get_outputs(self, merge_multi_context=True):
        """Gets outputs of the previous forward computation. As a output loss module,
        we treat the inputs to this module as scores, and simply return them.

        Parameters
        ----------
        merge_multi_context : bool
            Should always be ``True``, because we do not use multiple contexts for computing.
        """
        assert merge_multi_context is True
        return [self._scores]

    def backward(self, out_grads=None):
        """Backward computation.

        Parameters
        ----------
        out_grads : NDArray or list of NDArray, optional
            Gradient on the outputs to be propagated back.
            This parameter is only needed when bind is called
            on outputs that are not a loss function.
        """
        assert out_grads is None, 'For a loss module, out_grads should be None'
        assert self.for_training

        self._backward_impl()

    def _backward_impl(self):
        """Actual implementation of the backward computation. The computation
        should take ``self._scores`` and ``self._labels`` and then compute the
        gradients with respect to the scores, store it as an `NDArray` in
        ``self._scores_grad``.

        Instead of defining a subclass and overriding this function,
        a more convenient way is to pass in a `grad_func` when constructing
        the module object. Then it will be called to compute the gradients.
        """
        if self._grad_func is not None:
            grad = self._grad_func(self._scores, self._labels)
            if not isinstance(grad, nd.NDArray):
                grad = nd.array(grad)
            self._scores_grad = grad
        else:
            raise NotImplementedError()

    def get_input_grads(self, merge_multi_context=True):
        """Gets the gradients to the inputs, computed in the previous backward computation.

        Parameters
        ----------
        merge_multi_context : bool
            Should always be ``True`` because we do not use multiple context for computation.
        """
        assert merge_multi_context is True
        return [self._scores_grad]

    def install_monitor(self, mon):
        """Installs monitor on all executors."""
        raise NotImplementedError()
