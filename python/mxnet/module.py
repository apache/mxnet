"""A module is like a FeedForward model. but we would like to make it
easier to be composed. So it is more like the Torch modules.
"""

from . import context as ctx
from . import ndarray as nd
from . import optimizer as opt

from . import metric

from .executor_group import DataParallelExecutorGroup
from .model import _create_kvstore, _initialize_kvstore, _update_params, _update_params_on_kvstore
from .model import BatchEndParam
from .initializer import Uniform

import logging
import time
import numpy as np

def _as_list(obj):
    """A utility function that treat the argument as a list.

    Parameters
    ----------
    obj : object

    Returns
    -------
    If `obj` is a list, return it. Otherwise, return `[obj]` as a single-element list.
    """
    if isinstance(obj, list):
        return obj
    else:
        return [obj]


class BaseModule(object):
    """The base class of a modules. A module represents a computation component. The users
    could use imperative API to operate a module. Like calling `forward` with data, and collect
    the outputs. The basic module simply encapsulates a symbolic computation graph and its
    associated executors. More complicated modules might be composition of other modules.

    A module should have the following properties

    - `binded`: `bool`, indicating whether the memory buffers needed for computation
       has been allocated.
    - `for_training`: whether the module is binded for training (if binded).
    - `params_initialized`: `bool`, indicating whether the parameters of this modules
       has been initialized.
    - `optimizer_initialized`: 'bool`, indicating whether an optimizer is defined
       and initialized.
    - `symbol`: `Symbol`, the underlying symbolic computation graph, if exists.
      This property is not necessarily constant. For example, for `BucketingModule`,
      this property is simply the *current* symbol being used. For other modules,
      this value might not be well defined.
    - `arg_params`: `dict` of name to `NDArray` mapping for module parameters.
    - `aux_params`: `dict` of name to `NDArray` mapping for auxiliary variables.
    - `inputs_need_grad`: `bool`, indicating whether gradients with respect to the
      input data is needed. Might be useful when implementing composition of modules.

    A sub-class should implement the following intermediate-level APIs

    - `bind`: allocate memory and prepare environments for computation
    - `init_params`: initialize module parameters
    - `init_optimizer`: install optimizer for parameter updating
    - `forward`: forward computation
    - `get_outputs`: get the outputs of the previous forward computation
    - `backward`: backward computation
    - `update`: update parameters according to installed optimizer
    - `update_metric`: update evaluation metric

    When those intermediate-level API are implemented properly, the following
    high-level API will be automatically available for a module:

    - `fit`: train the module parameters on a data set
    - `predict`: run prediction on a data set and collect outputs
    - `score`: run prediction on a data set and evaluate performance

    Notes
    -----
    If you want to read the module parameters, they could be accessed directly from
    `module.arg_params`.  Those are `NDArray` that lives on CPU, regardless of the
    actual computation devices used. So if you updated the parameters on the devices
    using `module.update()` function, remember to call `module.sync_params_from_devices()`
    to sync the parameters back before reading `module.arg_params`.

    If you need to set the parameters. Do **not** assign values directly to
    `module.arg_params`. As they will not get propagated to the devices automatically.
    Instead, the function `init_params` should always be used for this purpose,
    potentially with `force_init=True`.
    """
    def __init__(self, logger=logging):
        self.logger = logger
        self.binded = False
        self.for_training = False
        self.inputs_need_grad = False
        self.params_initialized = False
        self.optimizer_initialized = False

    def forward_backward(self, data_batch, is_train=None):
        """A convenient function that calls both `forward` and `backward`.
        """
        self.forward(data_batch, is_train=is_train)
        self.backward()

    def score(self, eval_data, eval_metric, num_batch=None, batch_end_callback=None,
              reset=True, epoch=0):
        """Run prediction on `eval_data` and evaluate the performance according to
        `eval_metric`.

        Parameters
        ----------
        eval_data : DataIter
        eval_metric : EvalMetric
        num_batch : int
            Number of batches to run. Default is `None`, indicating run until the `DataIter`
            finishes.
        batch_end_callback : function
            Could also be a list of functions.
        reset : bool
            Default `True`, indicating whether we should reset `eval_data` before starting
            evaluating.
        epoch : int
            Default 0. For compatibility, this will be passed to callbacks (if any). During
            training, this will correspond to the training epoch number.
        """
        assert self.binded and self.params_initialized

        if reset:
            eval_data.reset()

        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        eval_metric.reset()
        for nbatch, eval_batch in enumerate(eval_data):
            if num_batch is not None and nbatch == num_batch:
                break

            self.forward(eval_batch, is_train=False)
            self.update_metric(eval_metric, eval_batch.label)

            if batch_end_callback is not None:
                batch_end_params = BatchEndParam(epoch=epoch,
                                                 nbatch=nbatch,
                                                 eval_metric=eval_metric,
                                                 locals=locals())
                for cb in _as_list(batch_end_callback):
                    cb(batch_end_params)

    def predict(self, eval_data, num_batch=None, merge_batches=True, reset=True,
                always_output_list=False):
        """Run prediction and collect the outputs.

        Parameters
        ----------
        eval_data : DataIter
        num_batch : int
            Default is `None`, indicating run all the batches in the data iterator.
        merge_batches : bool
            Default is `True`, see the doc for return values.
        reset : bool
            Default is `True`, indicating whether we should reset the data iter before start
            doing prediction.
        always_output_list : bool
            Default is `False`, see the doc for return values.

        Returns
        -------
        When `merge_batches` is `True` (by default), the return value will be a list
        `[out1, out2, out3]`.  Where each element is concatenation of the outputs for
        all the mini-batches. If further that `always_output_list` is `False` (by default),
        then in the case of a single output, `out1` is returned instead of `[out1]`.

        When `merge_batches` is `False`, the return value will be a nested list like
        `[[out1_batch1, out2_batch1], [out1_batch2], ...]`. This mode is useful because
        in some cases (e.g. bucketing), the module does not necessarily produce the same
        number of outputs.
        """
        assert self.binded and self.params_initialized

        if reset:
            eval_data.reset()

        output_list = []

        for nbatch, eval_batch in enumerate(eval_data):
            if num_batch is not None and nbatch == num_batch:
                break
            self.forward(eval_batch, is_train=False)
            pad = eval_batch.pad
            output_list.append(self.get_outputs())

        if len(output_list) == 0:
            return output_list

        if merge_batches:
            num_outputs = len(output_list[0])
            for o in output_list:
                assert len(o) == num_outputs, \
                       'Cannot merge batches, as num of outputs is not the same ' + \
                       'in mini-batches. Maybe bucketing is used?'
            output_list2 = [np.concatenate([o[i] for o in output_list])
                            for i in range(num_outputs)]

            if num_outputs == 1:
                return output_list2[0]
            return output_list2

        return output_list

    def fit(self, train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params={}, eval_batch_end_callback=None,
            initializer=Uniform(0.01), arg_params=None, aux_params=None,
            allow_missing=False, force_init=False, begin_epoch=0, num_epoch=None):
        """Train the module parameters.

        Parameters
        ----------
        train_data : DataIter
        eval_data : DataIter
            If not `None`, will be used as validation set and evaluate the performance
            after each epoch.
        eval_metric : str or EvalMetric
            Default `'acc'`. The performance measure used to display during training.
        epoch_end_callback : function or list of function
            Each callback will be called with the current `epoch`, `symbol`, `arg_params`
            and `aux_params`.
        batch_end_callback : function or list of function
            Each callback will be called with a `BatchEndParam`.
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `{}`. The parameters for the optimizer constructor.
        eval_batch_end_callback : function or list of function
        initializer : Initializer
            Will be called to initialize the module parameters if not already initialized.
        arg_params : dict
            Default `None`, if not `None`, should be existing parameters from a trained
            model or loaded from a checkpoint (previously saved model). In this case,
            the value here will be used to initialize the module parameters, unless they
            are already initialized by the user via a call to `init_params` or `fit`.
            `arg_params` has higher priority to `initializer`.
        aux_params : dict
            Default `None`. Similar to `arg_params`, except for auxiliary states.
        allow_missing : bool
            Default `False`. Indicate whether we allow missing parameters when `arg_params`
            and `aux_params` are not `None`. If this is `True`, then the missing parameters
            will be initialized via the `initializer`.
        force_init : bool
            Default `False`. Indicate whether we should force initialization even if the
            parameters are already initialized.
        begin_epoch : int
            Default `0`. Indicate the starting epoch. Usually, if we are resuming from a
            checkpoint saved at a previous training phase at epoch N, then we should specify
            this value as N+1.
        num_epoch : int
            Number of epochs to run training.
        """

        assert num_epoch is not None, 'please specify number of epochs'

        self.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
                  for_training=True, force_rebind=True)
        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)
        self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)

        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        ################################################################################
        # training loop
        ################################################################################
        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            eval_metric.reset()
            for nbatch, data_batch in enumerate(train_data):
                self.forward_backward(data_batch, is_train=True)
                self.update()
                self.update_metric(eval_metric, data_batch.label)

                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    for cb in _as_list(batch_end_callback):
                        cb(batch_end_params)

            # one epoch of training is finished
            for name, val in eval_metric.get_name_value():
                self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            # sync parameters back to CPU
            if epoch_end_callback or epoch+1 == num_epoch:
                self.sync_params_from_devices()

            if epoch_end_callback is not None:
                for cb in _as_list(epoch_end_callback):
                    cb(epoch, self.symbol, self.arg_params, self.aux_params)

            #----------------------------------------
            # evaluation on validation set
            if eval_data:
                self.score(eval_data, eval_metric,
                           batch_end_callback=eval_batch_end_callback, epoch=epoch)
                for name, val in eval_metric.get_name_value():
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

            # end of 1 epoch, reset the data-iter for another epoch
            train_data.reset()


class Module(BaseModule):
    """Module is a basic module that wrap a `Symbol`. It is functionally the same
    as the `FeedForward` model, except under the module API.

    Parameters
    ----------
    symbol : Symbol
    input_names : list of string
        Default is `['data', 'softmax_label']` for a typical model used in
        image classification.
    logger : Logger
        Default is `logging`.
    context : Context or list of Context
        Default is `cpu()`.
    work_load_list : list of number
        Default `None`, indicating uniform workload.
    """
    def __init__(self, symbol, input_names=['data', 'softmax_label'], logger=logging,
                 context=ctx.cpu(), work_load_list=None):
        super(Module, self).__init__(logger=logger)

        if isinstance(context, ctx.Context):
            context = [context]
        self.context = context
        if work_load_list is None:
            work_load_list = [1] * len(self.context)
        assert len(work_load_list) == len(self.context)
        self.work_load_list = work_load_list

        self.symbol = symbol

        arg_names = symbol.list_arguments()
        self.param_names = [x for x in arg_names if x not in input_names]
        self.aux_names = symbol.list_auxiliary_states()

        self.arg_params = None
        self.aux_params = None

        self._reset_bind()

    def _reset_bind(self):
        self.binded = False
        self.exec_group = None

    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None):
        """Bind the symbols to construct executors. This is necessary before one
        can perform computation with the module.

        Parameters
        ----------
        data_shapes : list of (str, tuple)
            Typically is `data_iter.provide_data`.
        label_shapes : list of (str, tuple)
            Typically is `data_iter.provide_label`.
        for_training : bool
            Default is `True`. Whether the executors should be bind for training.
        inputs_need_grad : bool
            Default is `False`. Whether the gradients to the input data need to be computed.
            Typically this is not needed. But this might be needed when implementing composition
            of modules.
        force_rebind : bool
            Default is `False`. This function does nothing if the executors are already
            binded. But with this `True`, the executors will be forced to rebind.
        shared_module : Module
            Default is `None`. This is used in bucketing. When not `None`, the shared module
            essentially corresponds to a different bucket -- a module with different symbol
            but with the same sets of parameters (e.g. unrolled RNNs with different lengths).
        """
        # force rebinding is typically used when one want to switch from
        # training to prediction phase.
        if force_rebind:
            self._reset_bind()

        if self.binded:
            self.logger.warning('Already binded, ignoring bind()')
            return

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad
        self.binded = True

        if not for_training:
            assert not inputs_need_grad
        else:
            assert label_shapes is not None

        if shared_module is not None:
            assert isinstance(shared_module, Module) and \
                    shared_module.binded and shared_module.params_initialized
            shared_group = shared_module.exec_group
        else:
            shared_group = None

        self.exec_group = DataParallelExecutorGroup(self.symbol, self.context, self.work_load_list,
                                                    data_shapes, label_shapes, self.param_names,
                                                    for_training, inputs_need_grad, shared_group)

        if shared_module is not None:
            self.params_initialized = True
            self.arg_params = shared_module.arg_params
            self.aux_params = shared_module.aux_params

        if self.params_initialized:
            # if the parameters are already initialized, we are re-binding
            # so automatically copy the already initialized params
            self.exec_group.set_params(self.arg_params, self.aux_params)

        if shared_module is not None and shared_module.optimizer_initialized:
            self.borrow_optimizer(shared_module)

    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False):
        """Initialize the parameters and auxiliary states.

        Parameters
        ----------
        initializer : Initializer
            Called to initialize parameters if needed.
        arg_params : dict
            If not None, should be a dictionary of existing arg_params. Initialization
            will be copied from that.
        aux_params : dict
            If not None, should be a dictionary of existing aux_params. Initialization
            will be copied from that.
        allow_missing : bool
            If true, params could contain missing values, and the initializer will be
            called to fill those missing params.
        force_init : bool
            If true, will force re-initialize even if already initialized.
        """
        if self.params_initialized and not force_init:
            return
        assert self.binded, 'call bind before initializing the parameters'

        if self.arg_params is None:
            param_arrays = [nd.zeros(x[0].shape) for x in self.exec_group.param_arrays]
            self.arg_params = {name:arr for name, arr in zip(self.param_names, param_arrays)}

        if self.aux_params is None:
            aux_arrays = [nd.zeros(x[0].shape) for x in self.exec_group.aux_arrays]
            self.aux_params = {name:arr for name, arr in zip(self.aux_names, aux_arrays)}

        def _impl(name, arr, cache):
            if cache is not None:
                if cache.has_key(name):
                    cache[name].copyto(arr)
                else:
                    assert allow_missing
                    initializer(name, arr)
            else:
                initializer(name, arr)

        for name, arr in self.arg_params.iteritems():
            _impl(name, arr, arg_params)

        for name, arr in self.aux_params.iteritems():
            _impl(name, arr, aux_params)

        self.params_initialized = True

        # copy the initialized parameters to devices
        self.exec_group.set_params(self.arg_params, self.aux_params)

    def forward(self, data_batch, is_train=None):
        """Forward computation.

        Parameters
        ----------
        data_batch : DataBatch
            Could be anything with similar API implemented.
        is_train : bool
            Default is `None`, which means `is_train` takes the value of `self.for_training`.
        """
        assert self.binded and self.params_initialized
        self.exec_group.forward(data_batch, is_train)

    def backward(self):
        """Backward computation.
        """
        assert self.binded and self.params_initialized
        self.exec_group.backward()

    def get_outputs(self, merge_multi_context=True):
        """Get outputs of the previous forward computation.

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
        outputs = [[exec_.outputs[i].asnumpy() for exec_ in self.exec_group.execs]
                   for i in range(len(self.exec_group.execs[0].outputs))]
        if merge_multi_context:
            outputs = [np.concatenate(x) for x in outputs]
        return outputs

    def init_optimizer(self, kvstore='local', optimizer='sgd', optimizer_params={},
                       force_init=False):
        """Install and initialize optimizers.

        Parameters
        ----------
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `{}`
        force_init : bool
            Default `False`, indicating whether we should force re-initializing the
            optimizer in the case an optimizer is already installed.
        """
        assert self.binded and self.params_initialized

        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring...')
            return

        (kvstore, update_on_kvstore) = \
                _create_kvstore(kvstore, len(self.context), self.arg_params)

        if isinstance(optimizer, str):
            batch_size = self.exec_group.batch_size
            if kvstore and kvstore.type == 'dist_sync':
                batch_size *= kvstore.num_workers
            optimizer = opt.create(optimizer,
                                   rescale_grad=(1.0/batch_size), **optimizer_params)
        else:
            assert isinstance(optimizer, opt.Optimizer)

        self.optimizer = optimizer
        self.kvstore = kvstore
        self.update_on_kvstore = update_on_kvstore
        self.updater = None

        if not update_on_kvstore:
            self.updater = opt.get_updater(optimizer)
        if kvstore:
            # copy initialized local parameters to kvstore
            _initialize_kvstore(kvstore=kvstore,
                                param_arrays=self.exec_group.param_arrays,
                                arg_params=self.arg_params,
                                param_names=self.param_names,
                                update_on_kvstore=update_on_kvstore)
        if update_on_kvstore:
            kvstore.set_optimizer(self.optimizer)

        self.optimizer_initialized = True

    def borrow_optimizer(self, shared_module):
        """Borrow optimizer from a shared module. Used in bucketing, where exactly the same
        optimizer (esp. kvstore) is used.

        Parameters
        ----------
        shared_module : Module
        """
        assert shared_module.optimizer_initialized
        self.optimizer = shared_module.optimizer
        self.kvstore = shared_module.kvstore
        self.update_on_kvstore = shared_module.update_on_kvstore
        self.updater = shared_module.updater
        self.optimizer_initialized = True

    def update(self):
        """Update parameters according to the installed optimizer and the gradients computed
        in the previous forward-backward batch.
        """
        assert self.binded and self.params_initialized and self.optimizer_initialized

        if self.update_on_kvstore:
            _update_params_on_kvstore(self.exec_group.param_arrays,
                                      self.exec_group.grad_arrays,
                                      self.kvstore)
        else:
            _update_params(self.exec_group.param_arrays,
                           self.exec_group.grad_arrays,
                           updater=self.updater,
                           num_device=len(self.context),
                           kvstore=self.kvstore)

    def update_metric(self, eval_metric, labels):
        """Evaluate and accumulate evaluation metric on outputs of the last forward computation.

        Parameters
        ----------
        eval_metric : EvalMetric
        labels : list of NDArray
            Typically `data_batch.label`.
        """
        self.exec_group.update_metric(eval_metric, labels)

    def sync_params_from_devices(self):
        """Synchronize parameters from devices to CPU. This function should be called after
        calling `update` that updates the parameters on the devices, before one can read the
        latest parameters from `self.arg_params` and `self.aux_params`.
        """
        self.exec_group.get_params(self.arg_params, self.aux_params)


class BucketingModule(BaseModule):
    """A bucketing module is a module that support bucketing.

    Parameters
    ----------
    sym_gen : function
        A function when called with a bucket key, return a symbol corresponding to
        that bucket key.
    default_bucket_key : str (or any python object)
        The key for the default bucket.
    default_input_names : list of str
        Input names (names of data and labels) for the default bucket.
    logger : Logger
    context : Context or list of Context
        Default `cpu()`
    work_load_list : list of number
        Default `None`, indicating uniform workload.
    """
    def __init__(self, sym_gen, default_bucket_key=None, default_input_names=None,
                 logger=logging, context=ctx.cpu(), work_load_list=None):
        super(BucketingModule, self).__init__(logger=logger)

        assert default_bucket_key is not None
        assert default_input_names is not None, 'please specify input names for the default bucket'
        self.default_bucket_key = default_bucket_key
        self.default_input_names = default_input_names

        self.sym_gen = sym_gen
        self.context = context
        self.work_load_list = work_load_list

        self._reset_bind()

    def _reset_bind(self):
        """Internal utility function to reset binding."""
        self.binded = False
        self.buckets = {}
        self.curr_module = None

    def _gen_symbol(self, key):
        """Internal utility function to generate symbol and collect input names."""
        assert self.binded
        symbol = self.sym_gen(key)
        arg_names = symbol.list_arguments()

        # we assume in the bucketing case, all symbols have the same set of parameters,
        # and all the rest of the arguments are considered as input names
        input_names = [x for x in arg_names if not x in self.curr_module.param_names]
        return symbol, input_names

    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False):
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
        """
        # force rebinding is typically used when one want to switch from
        # training to prediction phase.
        if force_rebind:
            self._reset_bind()

        if self.binded:
            self.logger.warning('Already binded, ignoring bind()')
            return

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad
        self.binded = True

        symbol = self.sym_gen(self.default_bucket_key)
        module = Module(symbol, self.default_input_names, logger=self.logger, context=self.context,
                        work_load_list=self.work_load_list)
        module.bind(data_shapes, label_shapes, for_training, inputs_need_grad,
                    force_rebind=False, shared_module=None)
        self.curr_module = module
        self.buckets[self.default_bucket_key] = module

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
        self.curr_module.init_params(initializer=initializer, arg_params=arg_params,
                                     aux_params=aux_params, allow_missing=allow_missing,
                                     force_init=force_init)
        self.params_initialized = True

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
        if not self.buckets.has_key(bucket_key):
            symbol, input_names = self._gen_symbol(bucket_key)
            module = Module(symbol, input_names, logger=self.logger, context=self.context,
                            work_load_list=self.work_load_list)
            module.bind(data_shapes, label_shapes, self.curr_module.for_training,
                        self.curr_module.inputs_need_grad,
                        force_rebind=False, shared_module=self.curr_module)
            self.buckets[bucket_key] = module

        self.curr_module = self.buckets[bucket_key]


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
        self.curr_module.forward(data_batch, is_train=is_train)

    def backward(self):
        """Backward computation."""
        self.curr_module.backward()

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
        return self.curr_module.get_outputs(merge_multi_context=merge_multi_context)

    def init_optimizer(self, kvstore='local', optimizer='sgd', optimizer_params=dict(),
                       force_init=False):
        """Install and initialize optimizers.

        Parameters
        ----------
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `{}`
        force_init : bool
            Default `False`, indicating whether we should force re-initializing the
            optimizer in the case an optimizer is already installed.
        """
        assert self.binded and self.params_initialized
        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring.')
            return

        self.curr_module.init_optimizer(kvstore, optimizer, optimizer_params, force_init=force_init)
        for mod in self.buckets.itervalues():
            if mod is not self.curr_module:
                mod.borrow_optimizer(self.curr_module)

        self.optimizer_initialized = True

    def update(self):
        """Update parameters according to installed optimizer and the gradient computed
        in the previous forward-backward cycle.
        """
        assert self.binded and self.params_initialized and self.optimizer_initialized
        self.curr_module.update()

    def update_metric(self, eval_metric, labels):
        """Evaluate and accumulate evaluation metric on outputs of the last forward computation.

        Parameters
        ----------
        eval_metric : EvalMetric
        labels : list of NDArray
            Typically `data_batch.label`.
        """
        self.curr_module.update_metric(eval_metric, labels)

    def sync_params_from_devices(self):
        """Synchronize parameters from devices to CPU. This function should be called after
        calling `update` that updates the parameters on the devices, before one can read
        the latest parameters from `self.arg_params` and `self.aux_params`.
        """
        self.curr_module.sync_params_from_devices()

    @property
    def arg_params(self):
        """Parameters for the current module."""
        assert self.binded and self.params_initialized
        return self.curr_module.arg_params

    @property
    def aux_params(self):
        """Auxiliary states of the current module."""
        assert self.binded and self.params_initialized
        return self.curr_module.aux_params

    @property
    def symbol(self):
        """The symbol of the current bucket being used."""
        assert self.binded
        return self.curr_module.symbol
