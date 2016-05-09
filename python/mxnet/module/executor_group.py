# pylint: disable=too-many-instance-attributes,too-many-locals
# pylint: disable=too-many-branches,too-many-statements,too-many-arguments
"""Executor group is a convenient tool for managing a group of executors."""

import numpy as np
import logging

from .. import context as ctx
from .. import ndarray as nd

from ..base import mx_real_t
from ..executor_manager import _split_input_slice, _load_data, _load_label

def _merge_multi_context(outputs):
    """Merge outputs that lives on multiple context into one, so that they look
    like living on one context.
    """
    outputs = [nd.concatenate(x, always_copy=False) for x in outputs]
    return outputs

class DataParallelExecutorGroup(object):
    """DataParallelExecutorGroup is a group of executors that lives on a group of devices.
    This is a helper class used to implement data parallelization. Each mini-batch will
    be split and run on the devices.

    Parameters
    ----------
    symbol : Symbol
        The common symbolic computation graph for all executors.
    contexts : list
        A list of contexts.
    workload : list
        If not `None`, could be a list of numbers that specify the workload to be assigned
        to different context. Larger number indicate heavier workload.
    data_shapes : list
        Should be a list of (name, shape) tuples, for the shapes of data. Note the order is
        important and should be the same as the order that the `DataIter` provide the data.
    label_shapes : list
        Should be a list of (name, shape) tuples, for the shapes of label. Note the order is
        important and should be the same as the order that the `DataIter` provide the label.
    param_names : list
        A list of strings, indicating the names of parameters (e.g. weights, filters, etc.)
        in the computation graph.
    for_training : bool
        Indicate whether the executors should be bind for training. When not doing training,
        the memory for gradients will not be allocated.
    inputs_need_grad : bool
        Indicate whether the gradients for the input data should be computed. This is currently
        not used. It will be useful for implementing composition of modules.
    shared_group : DataParallelExecutorGroup
        Default is `None`. This is used in bucketing. When not `None`, it should be a executor
        group corresponding to a different bucket. In other words, it will correspond to a different
        symbol but with the same set of parameters (e.g. unrolled RNNs with different lengths).
        In this case, many memory will be shared.
    input_types : list
        Default is `None`. When not `None`, can be used to specify the data type for each
        of the data/label inputs.
    logger : Logger
        Default is `logging`.
    """
    def __init__(self, symbol, contexts, workload, data_shapes, label_shapes, param_names,
                 for_training, inputs_need_grad, shared_group=None, input_types=None,
                 logger=logging):
        self.param_names = param_names
        self.arg_names = symbol.list_arguments()
        self.aux_names = symbol.list_auxiliary_states()

        self.symbol = symbol
        self.contexts = contexts
        self.workload = workload

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad

        self.input_types = input_types
        self.logger = logger

        if shared_group is not None:
            self.shared_data_arrays = shared_group.shared_data_arrays
        else:
            self.shared_data_arrays = [{} for _ in contexts]

        # initialize some instance variables
        self.batch_size = None
        self.slices = None
        self.execs = None
        self.data_arrays = None
        self.label_arrays = None
        self.param_arrays = None
        self.grad_arrays = None
        self.aux_arrays = None

        # calculate workload and bind executors
        self.decide_slices(data_shapes)
        self.bind_exec(data_shapes, label_shapes, shared_group)

    def decide_slices(self, data_shapes):
        """Decide the slices for each context according to the workload.

        Parameters
        ----------
        data_shapes : list
            list of (name, shape) specifying the shapes for the input data.
        """
        assert len(data_shapes) > 0
        self.batch_size = data_shapes[0][1][0]
        for shape in data_shapes:
            assert shape[1][0] == self.batch_size, "all the data must have the same batch size"

        self.slices = _split_input_slice(self.batch_size, self.workload)

    def bind_exec(self, data_shapes, label_shapes, shared_group):
        """Bind executors on their respective devices.

        Parameters
        ----------
        data_shapes : list
        label_shapes : list
        shared_group : DataParallelExecutorGroup
        """
        self.execs = []
        for i in range(len(self.contexts)):
            self.execs.append(self._bind_ith_exec(i, data_shapes, label_shapes, shared_group))

        # convenient data structures
        self.data_arrays = [[(self.slices[i], e.arg_dict[name]) for i, e in enumerate(self.execs)]
                            for name, _ in data_shapes]
        if label_shapes is not None:
            self.label_arrays = [[(self.slices[i], e.arg_dict[name])
                                  for i, e in enumerate(self.execs)]
                                 for name, _ in label_shapes]
        else:
            self.label_arrays = None

        self.param_arrays = [[exec_.arg_arrays[i] for exec_ in self.execs]
                             for i, name in enumerate(self.arg_names)
                             if name in self.param_names]
        if self.for_training:
            self.grad_arrays = [[exec_.grad_arrays[i] for exec_ in self.execs]
                                for i, name in enumerate(self.arg_names)
                                if name in self.param_names]
        else:
            self.grad_arrays = None

        data_names = [x[0] for x in data_shapes]
        if self.inputs_need_grad:
            self.input_grad_arrays = [[exec_.grad_arrays[i] for exec_ in self.execs]
                                      for i, name in enumerate(self.arg_names)
                                      if name in data_names]
        else:
            self.input_grad_arrays = None

        self.aux_arrays = [[exec_.aux_arrays[i] for exec_ in self.execs]
                           for i in range(len(self.aux_names))]

    def set_params(self, arg_params, aux_params):
        """Assign, i.e. copy parameters to all the executors.

        Parameters
        ----------
        arg_params : dict
            A dictionary of name to `NDArray` parameter mapping.
        aux_params : dict
            A dictionary of name to `NDArray` auxiliary variable mapping.
        """
        for exec_ in self.execs:
            exec_.copy_params_from(arg_params, aux_params)

    def get_params(self, arg_params, aux_params):
        """ Copy data from each executor to `arg_params` and `aux_params`.

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
            weight = sum(w.copyto(ctx.cpu()) for w in block) / len(block)
            weight.astype(arg_params[name].dtype).copyto(arg_params[name])
        for name, block in zip(self.aux_names, self.aux_arrays):
            weight = sum(w.copyto(ctx.cpu()) for w in block) / len(block)
            weight.astype(aux_params[name].dtype).copyto(aux_params[name])

    def forward(self, data_batch, is_train=None):
        """Split `data_batch` according to workload and run forward on each devices.

        Parameters
        ----------
        data_batch : DataBatch
            Or could be any object implementing similar interface.
        is_train : bool
            The hint for the backend, indicating whether we are during training phase.
            Default is `None`, then the value `self.for_training` will be used.
        Returns
        -------

        """
        _load_data(data_batch, self.data_arrays)
        if is_train is None:
            is_train = self.for_training

        if is_train:
            # It could be the case that even though we are binded for training, we
            # still do not have label arrays. For example, this could happen if we
            # are using a module without a loss function (the user will compute the
            # loss and gradients using some other ways), and we do not need the label
            # here.
            if self.label_arrays is not None:
                _load_label(data_batch, self.label_arrays)

        for exec_ in self.execs:
            exec_.forward(is_train=is_train)

    def get_output_shapes(self):
        """Get the shapes of the outputs."""
        outputs = self.execs[0].outputs
        shapes = [out.shape for out in outputs]
        shapes = [tuple([self.batch_size] + list(shape[1:])) for shape in shapes]
        return zip(self.symbol.list_outputs(), shapes)

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
        elements are `NDArray`.
        """
        outputs = [[exec_.outputs[i] for exec_ in self.execs]
                   for i in range(len(self.execs[0].outputs))]
        if merge_multi_context:
            outputs = _merge_multi_context(outputs)
        return outputs

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
        assert self.inputs_need_grad
        if merge_multi_context:
            return _merge_multi_context(self.input_grad_arrays)
        return self.input_grad_arrays

    def backward(self, out_grads=None):
        """Run backward on all devices. A backward should be called after
        a call to the forward function. Backward cannot be called unless
        `self.for_training` is `True`.

        Parameters
        ----------
        out_grads : NDArray or list of NDArray, optional
            Gradient on the outputs to be propagated back.
            This parameter is only needed when bind is called
            on outputs that are not a loss function.
        """
        assert self.for_training, 're-bind with for_training=True to run backward'
        if out_grads is None:
            out_grads = []

        for i, (exec_, islice) in enumerate(zip(self.execs, self.slices)):
            out_grads_slice = [grad[islice].as_in_context(self.contexts[i])
                               for grad in out_grads]
            exec_.backward(out_grads=out_grads_slice)

    def update_metric(self, eval_metric, labels):
        """Accumulate the performance according to `eval_metric` on all devices.

        Parameters
        ----------
        eval_metric : EvalMetric
            The metric used for evaluation.
        labels : list of NDArray
            Typically comes from `label` of a `DataBatch`.
        """
        for texec, islice in zip(self.execs, self.slices):
            labels_slice = [label[islice] for label in labels]
            eval_metric.update(labels_slice, texec.outputs)

    def _bind_ith_exec(self, i, data_shapes, label_shapes, shared_group):
        """Internal utility function to bind the i-th executor.
        """
        data_shapes = self._sliced_shape(data_shapes, i)
        if label_shapes is not None:
            label_shapes = self._sliced_shape(label_shapes, i)
        shared_exec = None if shared_group is None else shared_group.execs[i]
        context = self.contexts[i]
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
                grad_req[name] = 'null'

        def _get_or_reshape(name, shared_data_arrays, arg_shape, arg_type, context, logger):
            """Internal helper to get a memory block or re-use by re-shaping"""
            if name in shared_data_arrays:
                arg_arr = shared_data_arrays[name]

                if np.prod(arg_arr.shape) >= np.prod(arg_shape):
                    # nice, we can directly re-use this data blob
                    assert arg_arr.dtype == arg_type
                    arg_arr = arg_arr.reshape(arg_shape)
                else:
                    logger.warning(('bucketing: data "%s" has a shape %s' % (name, arg_shape)) +
                                   (', which is larger than already allocated ') +
                                   ('shape %s' % (arg_arr.shape,)) +
                                   ('. Need to re-allocate. Consider putting ') +
                                   ('default_bucket_key to') +
                                   (' be the bucket taking the largest input for better ') +
                                   ('memory sharing.'))
                    arg_arr = nd.zeros(arg_shape, context, dtype=arg_type)

                    # replace existing shared array because the new one is bigger
                    shared_data_arrays[name] = arg_arr
            else:
                arg_arr = nd.zeros(arg_shape, context, dtype=arg_type)
                shared_data_arrays[name] = arg_arr

            return arg_arr

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
                    assert arg_arr.shape == arg_shapes[j]
                    assert arg_arr.dtype == arg_types[j]
                    if grad_req[name] != 'null':
                        grad_arrays[name] = shared_exec.grad_dict[name]
            else: # data or label
                arg_arr = _get_or_reshape(name, shared_data_arrays, arg_shapes[j], arg_types[j],
                                          context, self.logger)

                # data might also need grad if inputs_need_grad is True
                if grad_req[name] != 'null':
                    grad_arrays[name] = _get_or_reshape('grad of ' + name, shared_data_arrays,
                                                        arg_shapes[j], arg_types[j], context,
                                                        self.logger)

            arg_arrays.append(arg_arr)

        # create or borrow aux variables
        if shared_exec is None:
            aux_arrays = [nd.zeros(s, context, dtype=t) for s, t in zip(aux_shapes, aux_types)]
        else:
            for j, arr in enumerate(shared_exec.aux_arrays):
                assert aux_shapes[j] == arr.shape
                assert aux_types[j] == arr.dtype
            aux_arrays = shared_exec.aux_arrays[:]

        executor = self.symbol.bind(ctx=context, args=arg_arrays,
                                    args_grad=grad_arrays, aux_states=aux_arrays,
                                    grad_req=grad_req, shared_exec=shared_exec)
        return executor

    def _sliced_shape(self, shapes, i):
        """Get the sliced shapes for the i-th executor.

        Parameters
        ----------
        shapes : list of (str, tuple)
            The original (name, shape) pairs.
        i : int
            Which executor we are dealing with.
        """
        return [(k, tuple([self.slices[i].stop-self.slices[i].start] + list(v[1:])))
                for k, v in shapes]
