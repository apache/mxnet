# pylint: disable=too-many-instance-attributes, too-many-arguments
"""A `BucketingModule` implement the `BaseModule` API, and allows multiple
symbols to be used depending on the `bucket_key` provided by each different
mini-batch of data.
"""

import logging

from .. import context as ctx

from ..initializer import Uniform

from .base_module import BaseModule
from .module import Module

class BucketingModule(BaseModule):
    """A bucketing module is a module that support bucketing.

    Parameters
    ----------
    sym_gen : function
        A function when called with a bucket key, returns a triple
        `(symbol, data_names, label_names)`.
    default_bucket_key : str (or any python object)
        The key for the default bucket.
    logger : Logger
    context : Context or list of Context
        Default `cpu()`
    work_load_list : list of number
        Default `None`, indicating uniform workload.
    """
    def __init__(self, sym_gen, default_bucket_key=None,
                 logger=logging, context=ctx.cpu(), work_load_list=None):
        super(BucketingModule, self).__init__(logger=logger)

        assert default_bucket_key is not None
        self._default_bucket_key = default_bucket_key

        self._sym_gen = sym_gen
        self._context = context
        self._work_load_list = work_load_list

        self._buckets = {}
        self._curr_module = None

    def _reset_bind(self):
        """Internal utility function to reset binding."""
        self.binded = False
        self._buckets = {}
        self._curr_module = None

    @property
    def data_names(self):
        """A list of names for data required by this module."""
        if self.binded:
            return self._curr_module.data_names
        else:
            _, data_names, _ = self._sym_gen(self._default_bucket_key)
            return data_names

    @property
    def output_names(self):
        """A list of names for the outputs of this module."""
        if self.binded:
            return self._curr_module.output_names
        else:
            symbol, _, _ = self._sym_gen(self._default_bucket_key)
            return symbol.list_outputs()

    @property
    def data_shapes(self):
        """Get data shapes.
        Returns
        -------
        A list of `(name, shape)` pairs.
        """
        assert self.binded
        return self._curr_module.data_shapes

    @property
    def label_shapes(self):
        """Get label shapes.
        Returns
        -------
        A list of `(name, shape)` pairs. The return value could be `None` if
        the module does not need labels, or if the module is not binded for
        training (in this case, label information is not available).
        """
        assert self.binded
        return self._curr_module.label_shapes

    @property
    def output_shapes(self):
        """Get output shapes.
        Returns
        -------
        A list of `(name, shape)` pairs.
        """
        assert self.binded
        return self._curr_module.output_shapes

    def get_params(self):
        """Get current parameters.
        Returns
        -------
        `(arg_params, aux_params)`, each a dictionary of name to parameters (in
        `NDArray`) mapping.
        """
        assert self.binded and self.params_initialized
        return self._curr_module.get_params()

    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False):
        """Initialize parameters.

        Parameters
        ----------
        initializer : Initializer
        arg_params : dict
            Default `None`. Existing parameters. This has higher priority than `initializer`.
        aux_params : dict
            Default `None`. Existing auxiliary states. This has higher priority than `initializer`.
        allow_missing : bool
            Allow missing values in `arg_params` and `aux_params` (if not `None`). In this case,
            missing values will be filled with `initializer`.
        force_init : bool
            Default `False`.
        """
        if self.params_initialized and not force_init:
            return
        assert self.binded, 'call bind before initializing the parameters'
        self._curr_module.init_params(initializer=initializer, arg_params=arg_params,
                                      aux_params=aux_params, allow_missing=allow_missing,
                                      force_init=force_init)
        self.params_initialized = True

    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None,
             grad_req='write'):
        """Binding for a `BucketingModule` means setting up the buckets and bind the
        executor for the default bucket key. Executors corresponding to other keys are
        binded afterwards with `switch_bucket`.

        Parameters
        ----------
        data_shapes : list of (str, tuple)
            This should correspond to the symbol for the default bucket.
        label_shapes : list of (str, tuple)
            This should correspond to the symbol for the default bucket.
        for_training : bool
            Default is `True`.
        inputs_need_grad : bool
            Default is `False`.
        force_rebind : bool
            Default is `False`.
        shared_module : BucketingModule
            Default is `None`. This value is currently not used.
        grad_req : str, list of str, dict of str to str
            Requirement for gradient accumulation. Can be 'write', 'add', or 'null'
            (default to 'write').
            Can be specified globally (str) or for each argument (list, dict).
        """
        # in case we already initialized params, keep it
        if self.params_initialized:
            arg_params, aux_params = self.get_params()

        # force rebinding is typically used when one want to switch from
        # training to prediction phase.
        if force_rebind:
            self._reset_bind()

        if self.binded:
            self.logger.warning('Already binded, ignoring bind()')
            return

        assert shared_module is None, 'shared_module for BucketingModule is not supported'

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad
        self.binded = True

        symbol, data_names, label_names = self._sym_gen(self._default_bucket_key)
        module = Module(symbol, data_names, label_names, logger=self.logger,
                        context=self._context, work_load_list=self._work_load_list)
        module.bind(data_shapes, label_shapes, for_training, inputs_need_grad,
                    force_rebind=False, shared_module=None, grad_req=grad_req)
        self._curr_module = module
        self._buckets[self._default_bucket_key] = module

        # copy back saved params, if already initialized
        if self.params_initialized:
            self.set_params(arg_params, aux_params)

    def switch_bucket(self, bucket_key, data_shapes, label_shapes=None):
        """Switch to a different bucket. This will change `self.curr_module`.

        Parameters
        ----------
        bucket_key : str (or any python object)
            The key of the target bucket.
        data_shapes : list of (str, tuple)
            Typically `data_batch.provide_data`.
        label_shapes : list of (str, tuple)
            Typically `data_batch.provide_label`.
        """
        assert self.binded, 'call bind before switching bucket'
        if not bucket_key in self._buckets:
            symbol, data_names, label_names = self._sym_gen(bucket_key)
            module = Module(symbol, data_names, label_names,
                            logger=self.logger, context=self._context,
                            work_load_list=self._work_load_list)
            module.bind(data_shapes, label_shapes, self._curr_module.for_training,
                        self._curr_module.inputs_need_grad,
                        force_rebind=False, shared_module=self._curr_module)
            self._buckets[bucket_key] = module

        self._curr_module = self._buckets[bucket_key]

    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),),
                       force_init=False):
        """Install and initialize optimizers.

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
        assert self.binded and self.params_initialized
        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring.')
            return

        self._curr_module.init_optimizer(kvstore, optimizer, optimizer_params,
                                         force_init=force_init)
        for mod in self._buckets.values():
            if mod is not self._curr_module:
                mod.borrow_optimizer(self._curr_module)

        self.optimizer_initialized = True

    def forward(self, data_batch, is_train=None):
        """Forward computation.

        Parameters
        ----------
        data_batch : DataBatch
        is_train : bool
            Default is `None`, in which case `is_train` is take as `self.for_training`.
        """
        assert self.binded and self.params_initialized
        self.switch_bucket(data_batch.bucket_key, data_batch.provide_data,
                           data_batch.provide_label)
        self._curr_module.forward(data_batch, is_train=is_train)

    def backward(self, out_grads=None):
        """Backward computation."""
        assert self.binded and self.params_initialized
        self._curr_module.backward(out_grads=out_grads)

    def update(self):
        """Update parameters according to installed optimizer and the gradient computed
        in the previous forward-backward cycle.
        """
        assert self.binded and self.params_initialized and self.optimizer_initialized
        self._curr_module.update()

    def get_outputs(self, merge_multi_context=True):
        """Get outputs from a previous forward computation.

        Parameters
        ----------
        merge_multi_context : bool
            Default is `True`. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A `True` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        If `merge_multi_context` is `True`, it is like `[out1, out2]`. Otherwise, it
        is like `[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]`. All the output
        elements are numpy arrays.
        """
        assert self.binded and self.params_initialized
        return self._curr_module.get_outputs(merge_multi_context=merge_multi_context)

    def get_input_grads(self, merge_multi_context=True):
        """Get the gradients with respect to the inputs of the module.

        Parameters
        ----------
        merge_multi_context : bool
            Default is `True`. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A `True` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        If `merge_multi_context` is `True`, it is like `[grad1, grad2]`. Otherwise, it
        is like `[[grad1_dev1, grad1_dev2], [grad2_dev1, grad2_dev2]]`. All the output
        elements are `NDArray`.
        """
        assert self.binded and self.params_initialized and self.inputs_need_grad
        return self._curr_module.get_input_grads(merge_multi_context=merge_multi_context)

    def update_metric(self, eval_metric, labels):
        """Evaluate and accumulate evaluation metric on outputs of the last forward computation.

        Parameters
        ----------
        eval_metric : EvalMetric
        labels : list of NDArray
            Typically `data_batch.label`.
        """
        assert self.binded and self.params_initialized
        self._curr_module.update_metric(eval_metric, labels)

    @property
    def symbol(self):
        """The symbol of the current bucket being used."""
        assert self.binded
        return self._curr_module.symbol

    def install_monitor(self, mon):
        """ Install monitor on all executors """
        assert self.binded
        for mod in self._buckets.values():
            mod.install_monitor(mon)
