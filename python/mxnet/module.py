# A module is like a FeedForward, but we would like to make it
# easier to be composed. So it is more like the Torch modules.

from . import context as ctx
from . import symbol as sym
from . import ndarray as nd
from . import optimizer as opt

from .executor_manager import _split_input_slice, _load_data, _load_label
from .model import _create_kvstore, _initialize_kvstore, _update_params, _update_params_on_kvstore
from .base import mx_real_t
from .initializer import Uniform

from collections import namedtuple
import logging

class DataParallelExecutorGroup(object):
    def __init__(self, symbol, context, workload, data_shapes, label_shapes, param_names,
                 for_training, inputs_need_grad, shared_group=None, input_types=None):
        self.param_names = param_names
        self.arg_names = symbol.list_arguments()
        self.aux_names = symbol.list_auxiliary_states()

        self.symbol = symbol
        self.context = context
        self.workload = workload

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad

        self.input_types = input_types

        if shared_group is not None:
            self.shared_data_arrays = shared_group.shared_data_arrays
        else:
            self.shared_data_arrays = [{} for _ in context]

        self.decide_slices(data_shapes, label_shapes)
        self.bind_exec(data_shapes, label_shapes, shared_group)

    def decide_slices(self, data_shapes, label_shapes):
        assert len(data_shapes) > 0
        self.batch_size = data_shapes[0][1][0]
        for s in data_shapes:
            assert s[1][0] == self.batch_size, "all the data must have the same batch size"

        self.slices = _split_input_slice(self.batch_size, self.workload)

    def bind_exec(self, data_shapes, label_shapes, shared_group):
        self.execs = []
        for i in range(len(self.context)):
            self.execs.append(self._bind_ith_exec(i, data_shapes, label_shapes, shared_group))

        # convenient data structures
        self.data_arrays = [[(self.slices[i], e.arg_dict[name]) for i, e in enumerate(self.execs)]
                            for name, _ in data_shapes]
        if label_shapes is not None:
            self.label_arrays = [[(self.slices[i], e.arg_dict[name]) for i, e in enumerate(self.execs)]
                                 for name, _ in label_shapes]
        else:
            self.label_arrays = None


        self.param_arrays = [[e.arg_arrays[i] for e in self.execs]
                             for i, name in enumerate(self.arg_names)
                             if name in self.param_names]
        self.grad_arrays = [[e.grad_arrays[i] for e in self.execs]
                             for i, name in enumerate(self.arg_names)
                             if name in self.param_names]

        self.aux_arrays = [[e.aux_arrays[i] for e in self.execs]
                           for i in range(len(self.aux_names))]

    def set_params(self, arg_params, aux_params):
        for texec in self.execs:
            texec.copy_params_from(arg_params, aux_params)

    def forward(self, data_batch):
        _load_data(data_batch, self.data_arrays)
        if self.for_training:
            _load_label(data_batch, self.label_arrays)
        for texec in self.execs:
            texec.forward(is_train=self.for_training)

    def backward(self):
        assert self.for_training, 're-bind with for_training=True to run backward'
        for texec in self.execs:
            texec.backward()

    def update_metric(self, metric, labels):
        for texec, islice in zip(self.execs, self.slices):
            labels_slice = [label[islice] for label in labels]
            metric.update(labels_slice, texec.outputs)

    def _bind_ith_exec(self, i, data_shapes, label_shapes, shared_group):
        data_shapes = self._sliced_shape(data_shapes, i)
        if label_shapes is not None:
            label_shapes = self._sliced_shape(label_shapes, i)
        shared_exec = None if shared_group is None else shared_group.execs[i]
        context = self.context[i]
        shared_data_arrays = self.shared_data_arrays[i]

        input_shapes = dict(data_shapes)
        if label_shapes is not None:
            input_shapes.update(dict(label_shapes))

        arg_shapes, _, aux_shapes = self.symbol.infer_shape(**input_shapes)
        assert arg_shapes is not None, "shape inference failed"

        if self.input_types is None:
            input_types = {k: mx_real_t for k in input_shapes.keys()}
        else:
            input_types = self.input_types
        arg_types, _, aux_types = self.symbol.infer_type(**input_types)
        assert arg_types is not None, "type inference failed"

        data_names = [x[0] for x in data_shapes]

        arg_arrays = []
        grad_arrays = {} if self.for_training else None
        grad_req = {}
        for name in self.arg_names:
            if self.for_training:
                if name in self.param_names:
                    grad_req[name] = 'write'
                elif name in data_names:
                    grad_req[name] = 'write' if self.inputs_need_grad else 'null'
                else:
                    grad_req[name] = 'null'
            else:
                grad_req[name] == 'null'

        # create or borrow arguments and gradients
        for j in range(len(self.arg_names)):
            name = self.arg_names[j]
            if name in self.param_names: # model parameter
                if shared_exec is None:
                    arg_arr = nd.zeros(arg_shapes[j], context, dtype=arg_types[j])
                    if grad_req[name] != 'null':
                        grad_arr = nd.zeros(arg_shapes[j], context, dtype=arg_types[j])
                        grad_arrays[name] = grad_arr
                else:
                    arg_arr = shared_exec.arg_dict[name]
                    assert arg_arr.shape == arg_shape[j]
                    assert arg_arr.dtype == arg_types[j]
                    if grad_req[name] != 'null':
                        grad_arrays[name] = shared_exec.grad_dict[name]
            else: # data or label
                if name in shared_data_arrays:
                    arg_arr = shared_data_arrays[name]
                    assert arg_arr.shape == arg_shape[j]
                    assert arg_arr.dtype == arg_types[j]
                else:
                    arg_arr = nd.zeros(arg_shapes[j], context, dtype=arg_types[j])
                shared_data_arrays[name] = arg_arr

            arg_arrays.append(arg_arr)

        # create or borrow aux variables
        if shared_exec is None:
            aux_arrays = [nd.zeros(s, ctx, dtype=t) for s, t in zip(aux_shapes, aux_types)]
        else:
            for j, a in enumerate(shared_exec.aux_arrays):
                assert aux_shapes[j] == a.shape
                assert aux_types[j] == a.dtype
            aux_arrays = shared_exec.aux_arrays[:]

        executor = self.symbol.bind(ctx=context, args=arg_arrays,
                                    args_grad=grad_arrays, aux_states=aux_arrays,
                                    grad_req=grad_req, shared_exec=shared_exec)
        return executor

    def _sliced_shape(self, shapes, i):
        return [(k, tuple([self.slices[i].stop-self.slices[i].start] + list(v[1:])))
                for k, v in shapes]


class BaseModule(object):
    def __init__(self):
        self.binded = False
        self.params_initialized = False
        self.optimizer_initialized = False

    def train(self, train_data, valid_data):
        # TODO
        pass


class Module(BaseModule):
    def __init__(self, symbol, input_names, context=ctx.cpu(), work_load_list=None):
        super(Module, self).__init__()

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

        self._reset_bind()

    def _reset_bind(self):
        self.binded = False
        self.exec_group = None

    ################################################################################
    # === bind the module ===
    # binding a module allocate the memory required to carry out the computation
    # on the specific devices (contexts) specified.
    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_group=None):
        # force rebinding is typically used when one want to switch from
        # training to prediction phase.
        if force_rebind:
            self._reset_bind()

        if self.binded:
            logging.warning('Already binded, ignoring bind()')
            return

        self.for_training = for_training
        self.binded = True

        if not for_training:
            assert not inputs_need_grad
        else:
            assert label_shapes is not None

        self.exec_group = DataParallelExecutorGroup(self.symbol, self.context, self.work_load_list,
                                                    data_shapes, label_shapes, self.param_names,
                                                    for_training, inputs_need_grad, shared_group)
        if self.params_initialized:
            # if the parameters are already initialized, we are re-binding
            # so automatically copy the already initialized params
            self.exec_group.set_params(self.arg_params, self.aux_params)


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

        param_arrays = [nd.zeros(x[0].shape) for x in self.exec_group.param_arrays]
        self.arg_params = {name:arr for name, arr in zip(self.param_names, param_arrays)}

        aux_arrays = [nd.zeros(x[0].shape) for x in self.exec_group.aux_arrays]
        self.aux_params = {name:arr for name, arr in zip(self.aux_names, aux_arrays)}

        def _impl(name, arr, cache):
            if cache is not None:
                if cache.has_key(name):
                    cache[name].copy_to(arr)
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

    def forward(self, data_batch):
        assert self.binded and self.params_initialized
        self.exec_group.forward(data_batch)

    def backward(self):
        self.exec_group.backward()

    def init_optimizer(self, kvstore='local', optimizer='sgd', optimizer_params={},
                        force_init=False):
        assert self.binded and self.params_initialized

        if self.optimizer_initialized and not force_init:
            logging.warning('optimizer already initialized, ignoring...')
            return

        (kvstore, update_on_kvstore) = _create_kvstore(
                kvstore, len(self.context), self.arg_params)

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

    def update(self):
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

    def update_metric(self, metric, labels):
        self.exec_group.update_metric(metric, labels)
